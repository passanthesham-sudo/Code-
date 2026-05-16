%% intialization 
clear; clc; close all;

% Simulation Parameters 
wavelength = 1.55;       
mesh_res   = 40;         

% Geometric Profiles 
r_core = 4.1;            
r_clad = 20.0;          

% Refractive Indices 
n_core = 1.4457;         % Core index
n_clad = 1.4440;         % Cladding index

fprintf('INITIALIZING PRODUCTION FEM OPTICAL FIBER SOLVER \n');

[nodes, elems, regions] = build_fiber_mesh(r_core, r_clad, mesh_res);

modes = compute_modes(nodes, elems, regions, r_core, n_core, n_clad, wavelength, ...
                      'num_modes', 4, 'n_guess', (n_core + n_clad)/2);

% Plot Dominant Fundamental Mode (HE11)
if ~isempty(modes)
    plot_fiber_mode_fields(modes(1), nodes, elems, r_core, 'Fundamental Fiber Mode (HE_{11})');
end


%%  GEOMETRY & MESH GENERATION FUNCTIONS
function [nodes, elems, regions] = build_fiber_mesh(r_core, r_clad, mesh_res)
    n_rings_core = max(round(mesh_res * 0.4), 12);
    n_rings_clad = max(mesh_res - n_rings_core, 18);
   
    r_rings = [linspace(0, r_core, n_rings_core+1), ...
               linspace(r_core + (r_clad-r_core)/n_rings_clad, r_clad, n_rings_clad)];
    r_rings = unique(r_rings);
   
    pts = [0, 0];
    for i = 2:numel(r_rings)
        r = r_rings(i);
        n_ang = max(round(2 * pi * r / (r_clad / mesh_res * 0.75)), 12);
        theta = linspace(0, 2*pi, n_ang+1)';
        theta(end) = [];
        pts = [pts; r * cos(theta), r * sin(theta)];
    end
   
    outer_nodes = find(abs(sqrt(pts(:,1).^2 + pts(:,2).^2) - r_clad) < 1e-5);
    ang_sort = atan2(pts(outer_nodes,2), pts(outer_nodes,1));
    [~, o] = sort(ang_sort); outer_nodes = outer_nodes(o);
    C = [outer_nodes, [outer_nodes(2:end); outer_nodes(1)]];
   
    try
        DT = delaunayTriangulation(pts, C);
    catch
        DT = delaunayTriangulation(pts);
    end
   
    nodes_all = DT.Points;
    elems_all = DT.ConnectivityList;
   
    % Explicit multi-column indexing enforces exact vector alignment
    cx_all = (nodes_all(elems_all(:,1),1) + nodes_all(elems_all(:,2),1) + nodes_all(elems_all(:,3),1)) / 3;
    cy_all = (nodes_all(elems_all(:,1),2) + nodes_all(elems_all(:,2),2) + nodes_all(elems_all(:,3),2)) / 3;
   
    keep = sqrt(cx_all.^2 + cy_all.^2) <= r_clad + 1e-5;
    elems_keep = elems_all(keep, :);
   
    used = unique(elems_keep(:));
    node_map = zeros(size(nodes_all,1), 1);
    node_map(used) = 1:numel(used);
    nodes = nodes_all(used, :);
    elems = node_map(elems_keep);
   
    Ne = size(elems,1);
    cx = (nodes(elems(:,1),1) + nodes(elems(:,2),1) + nodes(elems(:,3),1)) / 3;
    cy = (nodes(elems(:,1),2) + nodes(elems(:,2),2) + nodes(elems(:,3),2)) / 3;
    cr = sqrt(cx.^2 + cy.^2);
    regions = ones(Ne, 1);      
    regions(cr <= r_core) = 2;  
   
    fprintf('  Fiber Mesh: %d nodes, %d elements\n', size(nodes,1), Ne);
end

function [edges, elem2edge, edge_sign] = build_edge_table(elems)
    Ne = size(elems,1); local_pairs = [2 3; 3 1; 1 2]; all_edges = zeros(3*Ne, 2);
    for ie = 1:Ne, for k = 1:3, all_edges(3*(ie-1)+k,:) = elems(ie, local_pairs(k,:)); end, end
    [edges, ~, ic] = unique(sort(all_edges, 2), 'rows'); elem2edge = reshape(ic, 3, Ne)';
    edge_sign = zeros(Ne,3);
    for ie = 1:Ne
        for k = 1:3
            if all_edges(3*(ie-1)+k,1) < all_edges(3*(ie-1)+k,2)
                edge_sign(ie,k) = 1;
            else
                edge_sign(ie,k) = -1;
            end
        end
    end
end

function [B_mat, detJ, grads] = element_geometry(xy)
    x = xy(:,1); y = xy(:,2);
    B_mat = [x(2)-x(1), x(3)-x(1); y(2)-y(1), y(3)-y(1)]; detJ = det(B_mat);
    grads = [y(2)-y(3), x(3)-x(2); y(3)-y(1), x(1)-x(3); y(1)-y(2), x(2)-x(1)] / (2*abs(detJ));
end

function [qp, qw] = tri_quadrature()
    qp = [1/6, 1/6; 2/3, 1/6; 1/6, 2/3]; qw = [1/6; 1/6; 1/6];
end

function bnd = boundary_nodes(elems, ~)
    all_edges = [elems(:,[1,2]); elems(:,[2,3]); elems(:,[3,1])];
    [uniq, ~, ic] = unique(sort(all_edges, 2), 'rows'); cnt = accumarray(ic, 1);
    bnd_edges = uniq(cnt == 1, :); bnd = unique(bnd_edges(:));
end


%%  FEM CORE EQUATION SOLVER MATRIX ASSEMBLY
function modes = compute_modes(nodes, elems, regions, r_core, n_core, n_clad, wavelength, varargin)
    p = inputParser;
    addParameter(p, 'num_modes', 4, @isnumeric);
    addParameter(p, 'mu_r', 1.0, @isnumeric);
    addParameter(p, 'n_guess', [], @isnumeric);
    parse(p, varargin{:}); opts = p.Results;

    c0 = 2.99792458e14; omega = 2*pi*c0 / wavelength; k0 = omega / c0;
    Nn = size(nodes,1); Ne = size(elems,1);
    [edges, elem2edge, edge_sign] = build_edge_table(elems);
    Nedge = size(edges,1); Ndof = Nedge + Nn;

    nnz_est = 36 * Ne;
    Ai = zeros(nnz_est,1,'int32'); Aj = zeros(nnz_est,1,'int32'); Av = zeros(nnz_est,1,'like', 1j);
    Bi = zeros(nnz_est,1,'int32'); Bj = zeros(nnz_est,1,'int32'); Bv = zeros(nnz_est,1,'like', 1j);
    cnt_A = 0; cnt_B = 0; mu_r = opts.mu_r;

    for ie = 1:Ne
        nds = elems(ie,:); xy = nodes(nds,:);
        edg = elem2edge(ie,:); sgn = edge_sign(ie,:);
        [~, detJ, grads] = element_geometry(xy); area = abs(detJ) / 2;
        edge_len = [norm(xy(3,:)-xy(2,:)), norm(xy(1,:)-xy(3,:)), norm(xy(2,:)-xy(1,:))];
        [qp, qw] = tri_quadrature();
       
        eps_val = ones(size(qp,1), 1) * n_clad^2;
        for q = 1:size(qp,1)
            pt_xy = [1-qp(q,1)-qp(q,2), qp(q,1), qp(q,2)] * xy;
            if sqrt(pt_xy(1)^2 + pt_xy(2)^2) <= r_core + 1e-5
                eps_val(q) = n_core^2;
            end
        end
       
        [Ae_local, Be_local] = element_matrices_quad(xy, grads, area, edge_len, sgn, qp, qw, eps_val, mu_r, k0);
        local_dofs = [edg, (nds + Nedge)];
        for ii = 1:6
            for jj = 1:6
                gi = local_dofs(ii); gj = local_dofs(jj);
                aval = Ae_local(ii,jj); bval = Be_local(ii,jj);
                if aval ~= 0, cnt_A = cnt_A+1; Ai(cnt_A) = gi; Aj(cnt_A) = gj; Av(cnt_A) = aval; end
                if bval ~= 0, cnt_B = cnt_B+1; Bi(cnt_B) = gi; Bj(cnt_B) = gj; Bv(cnt_B) = bval; end
            end
        end
    end

    A = sparse(double(Ai(1:cnt_A)), double(Aj(1:cnt_A)), Av(1:cnt_A), Ndof, Ndof);
    B = sparse(double(Bi(1:cnt_B)), double(Bj(1:cnt_B)), Bv(1:cnt_B), Ndof, Ndof);

    bnd_nodes = boundary_nodes(elems, Nn); bnd_dofs = bnd_nodes + Nedge;
    free_dofs = setdiff(1:Ndof, bnd_dofs);
    Af = A(free_dofs, free_dofs); Bf = B(free_dofs, free_dofs);

    sigma = k0^2 * (opts.n_guess)^2;
    opts_eigs.tol = 1e-12; opts_eigs.maxit = 600;

    [Vf, D] = eigs(-Af, -Bf, opts.num_modes, sigma, opts_eigs);

    lams = diag(D); [~, ord] = sort(real(sqrt(lams)), 'descend'); lams = lams(ord); Vf = Vf(:,ord);
    modes = struct();
    for m = 1:opts.num_modes
        x_full = zeros(Ndof,1,'like',1j); x_full(free_dofs) = Vf(:,m);
        modes(m).Et_dof = x_full(1:Nedge); modes(m).Ez_dof = x_full(Nedge+1:end);
        modes(m).n_eff = sqrt(lams(m)) / k0;
        modes(m).elem2edge = elem2edge; modes(m).edge_sign = edge_sign;
    end
    disp('Computed Effective Refractive Indices (n_eff):'); disp([modes.n_eff].');
end

function [Ae, Be] = element_matrices_quad(xy, grads, area, edge_len, sgn, qp, qw, eps_val, mu_r, k0)
    nq = size(qp,1); Ae = zeros(6,6,'like',1j); Be = zeros(6,6,'like',1j);
    local_edges = [2 3; 3 1; 1 2]; mu_t = mu_r; mu_z = mu_r;
    for q = 1:nq
        xi = qp(q,1); eta = qp(q,2); w = qw(q) * 2 * area; lam = [1-xi-eta, xi, eta];
        eps = eps_val(q);
       
        Wt = zeros(3,2); curlW = zeros(3,1);
        for k = 1:3
            Wt(k,:) = sgn(k) * edge_len(k) * (lam(local_edges(k,1))*grads(local_edges(k,2),:) - lam(local_edges(k,2))*grads(local_edges(k,1),:));
            gi = grads(local_edges(k,1),:); gj = grads(local_edges(k,2),:);
            curlW(k) = sgn(k) * edge_len(k) * 2 * (gi(1)*gj(2) - gi(2)*gj(1));
        end
        Phi = lam; GPhi = grads;
        for ii = 1:6
            for jj = 1:6
                if ii <= 3 && jj <= 3
                    Ae(ii,jj) = Ae(ii,jj) + w * ((1/mu_z) * curlW(ii) * curlW(jj) / k0^2 - eps * dot(Wt(ii,:), Wt(jj,:)));
                    Be(ii,jj) = Be(ii,jj) - w * (1/mu_t) * dot(Wt(ii,:), Wt(jj,:)) / k0^2;
                elseif ii <= 3 && jj > 3
                    Ae(ii,jj) = Ae(ii,jj) + w * (1/mu_t) * dot(GPhi(jj-3,:), Wt(ii,:));
                elseif ii > 3 && jj <= 3
                    Ae(ii,jj) = Ae(ii,jj) + w * eps * dot(Wt(jj,:), GPhi(ii-3,:));
                else
                    Ae(ii,jj) = Ae(ii,jj) + w * (-eps * Phi(ii-3) * Phi(jj-3) * k0^2);
                end
            end
        end
    end
end


%%  RECONSTRUCTED ANALYTIC FIBER PLOTTING UTILITY
function plot_fiber_mode_fields(mode, nodes, elems, r_core, ttl)
    Nn = size(nodes,1);
    Ne = size(elems,1);
    local_pairs = [2 3; 3 1; 1 2];
   
    Ex_n = zeros(Nn,1); Ey_n = zeros(Nn,1);
    Ez_n = abs(mode.Ez_dof);

    for ie = 1:Ne
        nds = elems(ie,:); xy = nodes(nds,:);
        edg = mode.elem2edge(ie,:); sgn = mode.edge_sign(ie,:);
        [~, detJ, grads] = element_geometry(xy);
        edge_len = [norm(xy(3,:)-xy(2,:)), norm(xy(1,:)-xy(3,:)), norm(xy(2,:)-xy(1,:))];
       
        for lnd = 1:3
            g_node = nds(lnd);
            lam = zeros(1,3); lam(lnd) = 1.0;
            Ex_q = 0; Ey_q = 0;
            for k = 1:3
                Wk = sgn(k)*edge_len(k)*(lam(local_pairs(k,1))*grads(local_pairs(k,2),:)-lam(local_pairs(k,2))*grads(local_pairs(k,1),:));
                Ex_q = Ex_q + mode.Et_dof(edg(k)) * Wk(1);
                Ey_q = Ey_q + mode.Et_dof(edg(k)) * Wk(2);
            end
            Ex_n(g_node) = Ex_n(g_node) + abs(Ex_q);
            Ey_n(g_node) = Ey_n(g_node) + abs(Ey_q);
        end
    end
   
    count = accumarray(elems(:), 1, [Nn 1]);
    Ex_n = Ex_n ./ count; Ey_n = Ey_n ./ count;
    E_transverse = sqrt(Ex_n.^2 + Ey_n.^2);
   
    E_transverse = E_transverse / max(E_transverse);
    Ez_n = Ez_n / max(Ez_n);

    figure('Name', ttl, 'Position', [100 100 1100 350]);
   
    th = linspace(0, 2*pi, 200);
    x_circle = r_core * cos(th);
    y_circle = r_core * sin(th);
   
    % Subplot 1: Transverse Profile Mode
    subplot(1,3,1);
    patch('Faces', elems, 'Vertices', nodes, 'FaceVertexCData', E_transverse, 'FaceColor', 'interp', 'EdgeColor', 'none');
    hold on; plot(x_circle, y_circle, 'w--', 'LineWidth', 1.5); hold off;
    title('|E_t| Transverse Mode Profile'); xlabel('x [\mum]'); ylabel('y [\mum]');
    colorbar; axis equal tight; colormap jet; xlim([-10 10]); ylim([-10 10]);

    % Subplot 2: Longitudinal Field Profile Mode
    subplot(1,3,2);
    patch('Faces', elems, 'Vertices', nodes, 'FaceVertexCData', Ez_n, 'FaceColor', 'interp', 'EdgeColor', 'none');
    hold on; plot(x_circle, y_circle, 'w--', 'LineWidth', 1.5); hold off;
    title('|E_z| Longitudinal Profile'); xlabel('x [\mum]'); ylabel('y [\mum]');
    colorbar; axis equal tight; xlim([-10 10]); ylim([-10 10]);

    % Subplot 3: Cross-Sectional Line Cut Check
    subplot(1,3,3);
    x_line = linspace(-10, 10, 200);
    E_cut = griddata(nodes(:,1), nodes(:,2), E_transverse, x_line, 0, 'cubic');
    plot(x_line, E_cut, 'b-', 'LineWidth', 2); hold on;
    xline(-r_core, 'r--', 'Core Boundary'); xline(r_core, 'r--');
    title('Mode Profile Center'); xlabel('Radius x [\mum]'); ylabel('Normalized Intensity');
    grid on; ylim([0 1.1]); xlim([-10 10]);

    sgtitle(sprintf('%s (n_{eff} = %.6f)', ttl, real(mode.n_eff)));
end