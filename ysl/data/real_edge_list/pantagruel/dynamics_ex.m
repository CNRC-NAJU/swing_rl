% The transient response of the continental European grid to
% an abrupt localized power loss is simulated in this example.
%
% System dynamics is governed by the following equations
% for conventional generators:
% 	m_i*do_i - d_i*o_i = p_i - sum_j b_ij*sin(t_i-t_j)
%
% for other buses:
%	d_i*o_i = p_i - sum_j b_ij*sin(t_i-t_j)
%
% where
% m_i: inertia
% d_i: primary control or load fred dep.
% p_i: power injection
% t_i: voltage phase
% o_i: d/dt t_i
% do_i: d/dt o_i 
% b_ij: line susceptance
% 
%   Author: Laurent Pagnier (laurent.vincent.pagnier@gmail.com)
%   December 14, 2019
%
% Requirements: Matlab (R2015b or later) and Matpower (5.0 or later). 

function dynamics_ex()
    clc
%     close all;
    dt = 5E-3;
    Ndt = 5E3;
    
    pantagruel = pantagruel_case;
    pantagruel.gen(:,10) = 0; % no minimal output
    
	disp('(This simulation might take some time)')
    disp(['   dt = ' num2str(dt)])
    Sb = pantagruel.baseMVA;
    N_bus = length(pantagruel.bus);
    N_line = length(pantagruel.branch);

    % run an OPF to dispatch the generation
	mpopt = mpoption('verbose',0,'out.all',0); % remove the OPF display
	mpopt.model = 'DC';
    
	pantagruel = runopf(pantagruel, mpopt);
    
    L = pantagruel.bus(:,3)/Sb;
    G = zeros(N_bus,1);
    % if a generator is not producing, it is not contributing to inertia
    % and primary control. Consequently, it is treated as a load bus.
    is_producing = pantagruel.gen(:,2) > 0;
    id_gen = pantagruel.gen(is_producing,1);
    id_load = setdiff(1:length(L),id_gen)';
    N_gen = length(id_gen);
    N_load = length(id_load);
    
    G(id_gen) = pantagruel.gen(is_producing,2)/Sb;
    P = -L + G;
    P = P - mean(P)*ones(size(P));

    % get the stable solution
    A = sparse([pantagruel.branch(:,1); pantagruel.branch(:,2)], [1:N_line 1:N_line], [ones(N_line,1);-ones(N_line,1)]); % incendence matrix
    b = -1i./pantagruel.branch(:,4);
    Ybus = conj(A*sparse(1:N_line, 1:N_line,b)*A');
    Q = zeros(N_bus,1);
    V = ones(N_bus,1);
    theta = zeros(N_bus,1);
    [~,theta,~,~] = NRsolver(Ybus, V, theta, -P, Q, [], 9, 1E-11, 1000);
    P_gen = P(id_gen);
    P_load = P(id_load);

    egdes = zeros(N_line,2);
    line_start = pantagruel.branch(:,1);
    line_end = pantagruel.branch(:,2);
    % bus reordering: generator buses first 
    for i=1:N_gen
        egdes(line_start == id_gen(i),1) = i;
        egdes(line_end == id_gen(i),2) = i;
    end
    for i=1:N_load
        egdes(line_start == id_load(i),1) = i + N_gen;
        egdes(line_end == id_load(i),2) = i + N_gen;
    end

    line_susceptance = 1./pantagruel.branch(:,4);

    % define the fault
    id_fault = length(id_gen)-3;
    dP = -900/Sb;
    P_gen(id_fault) = P_gen(id_fault) + dP;

    % define the dynamical parameters
    M_gen = pantagruel.gen_inertia(is_producing);
    D_gen = pantagruel.gen_prim_ctrl(is_producing) + pantagruel.load_freq_coef(id_gen);
    D_load = pantagruel.load_freq_coef(id_load);
    
	% set initial conditions
	omega_gen = zeros(N_gen,1);
	theta_gen = theta(id_gen);
	theta_load = theta(id_load);
    
    % directions are arbitrary
	incidence_mat = sparse([egdes(:,1);egdes(:,2)], [1:N_line 1:N_line], [ones(N_line,1);-ones(N_line,1)]);

    m = 10; % print a message and save frequencies every m iterations 
    omega_t = zeros(N_gen,floor(Ndt/m));
    k = 1;
    tic
    for i=1:Ndt
        y = radau5(omega_gen, theta_gen, theta_load, M_gen, D_gen, D_load, P_gen, P_load, incidence_mat, line_susceptance, dt, 14, 1E-6);
        if(mod(i,m)==0)
            disp(['   ' num2str(i) '/' num2str(Ndt) ' (' num2str(floor(100*i/Ndt)) '%)'])
            omega_t(:,k) = y(1:N_gen);
            k = k + 1;
        end
        omega_gen = y(1:N_gen);
        theta_gen = y(N_gen+1:2*N_gen);
        theta_load = y(2*N_gen+1:end);
    end
    toc
    figure
    plot(dt*m:dt*m:dt*m*floor(Ndt/m),omega_t')
    xlabel('time [s]')
    ylabel('frequency [s^{-1}]')
end

function [V,theta,iter,J] = NRsolver(Ybus, V, theta, P0, Q0, id_PQ, id_slack, epsilon, maxiter)
%=========================================================
%   For information on solving the power flow equations with 
%	Newton-Raphson, see, for instance,
%	V. Vittal and A. Bergen, Power systems analysis,
%   Prentice Hall, 1999.
%=========================================================
    if(nargin < 9)
        maxiter = 50;
    end
    n = length(Ybus);
    error  = 2*epsilon;
    iter = 0;  
    id = [1:id_slack-1 id_slack+1:n];
    while(error > epsilon && iter < maxiter)
        Vc = V.*exp(1i*theta);
        S = Vc.*(conj(Ybus)*conj(Vc));
        dPQ = [real(S(id))-P0(id); imag(S(id_PQ))-Q0(id_PQ)];
        temp1 = -1i*sparse(1:n,1:n,Vc)*conj(Ybus) * sparse(1:n,1:n,conj(Vc)) + 1i*sparse(1:n,1:n,Vc.*conj(Ybus*Vc));
        temp2 = sparse(1:n,1:n,Vc)*conj(Ybus) * sparse(1:n,1:n,exp(-1i*theta)) + sparse(1:n,1:n,exp(1i*theta).*conj(Ybus*Vc));
        J = [real(temp1(id,id)) real(temp2(id,id_PQ)); imag(temp1(id_PQ,id)) imag(temp2(id_PQ,id_PQ))];
        x = J\dPQ;
        theta(id) = theta(id) - x(1:n-1);
        if(~isempty(id_PQ))
            V(id_PQ) = V(id_PQ) - x(n:end);
        end
        error = max(abs(dPQ));
        iter = iter + 1;
    end
	if(iter == maxiter)
        disp(['Max iteration reached, error ' num2str(error)])
    end
end

function y=radau5(omega_gen, theta_gen, theta_load, M_gen, D_gen, D_load, P_gen, P_load, incidence_mat, line_susceptance, dt, maxiter, tol)
%=========================================================
%   For information on Radau methods, see, for instance,
%   E. Hairer and G. Wanner, Stiff differential equations solved by Radau
%   methods, J. Comput. Appl. Math. 11(1-2): 93-111 (1999)
%=========================================================

    % butcher tableau for radau 5 
    a = [(88-7*sqrt(6))/360, (296-169*sqrt(6))/1800, (-2+3*sqrt(6))/225
        (296+169*sqrt(6))/1800, (88+7*sqrt(6))/360, (-5-3*sqrt(6))/225
       (16-sqrt(6))/36, (16+sqrt(6))/36, 1/9];
    Ns = 3;
    
    % diagonalisation of A
    [Tr, lambda, Tl] = eig(inv(a));
    Tl = Tl';
    lambda = diag(lambda);
    d2 = diag(Tl*Tr);
    Tr(:,1) = Tr(:,1)/d2(1);
    Tr(:,2) = Tr(:,2)/d2(2);
    Tr(:,3) = Tr(:,3)/d2(3);
    
    y0 = [omega_gen; theta_gen; theta_load];
    
    N_line = length(line_susceptance);
    Ap = abs(incidence_mat);
    B = sparse(1:N_line, 1:N_line, line_susceptance);
    N_gen = length(omega_gen);
    N_bus = N_gen + length(theta_load);
    N_var = N_gen + N_bus;
    
    M1 = kron(Tr,sparse(1:N_var, 1:N_var,ones(N_var,1)));
    M2 = kron(diag(lambda)*Tl,sparse(1:N_var, 1:N_var,ones(N_var,1)));
    
    mat1 = incidence_mat(1:N_gen,:)*B;
    mat2 = incidence_mat(N_gen+1:end,:)*B;
    dtheta = incidence_mat'*y0(N_gen+1:end);
    
    % Jacobian matrix
    M = sparse(1:N_var, 1:N_var, [M_gen; ones(N_gen,1); D_load]);
    J0 = Ap * sparse(1:N_line, 1:N_line, line_susceptance .* cos(dtheta)) * Ap';
    J0 = J0 - sparse(1:N_bus, 1:N_bus,2*diag(J0));
    
    J = [sparse(1:N_gen, 1:N_gen, -D_gen) J0(1:N_gen,:);
        sparse(1:N_gen,1:N_gen,ones(N_gen,1)) sparse(N_gen,N_bus);
        sparse(N_bus-N_gen,N_gen) J0(N_gen+1:end,:)];

    P = [P_gen; zeros(N_gen,1); P_load];
   
    Y = repmat(y0,Ns,1);
    F = zeros(length(y0)*Ns, 1);
    W = zeros(length(y0)*Ns, 1);
    dW = zeros(length(y0)*Ns, 1);
    iter = 0;
    error = 2*tol;
    % Newton method
	while(error > tol && iter < maxiter)
        for s = 1:Ns
            F(N_var*(s-1)+1:N_var*s) = M*(Y(N_var*(s-1)+1:N_var*s)-y0);
            for o = 1:Ns
                dtheta = incidence_mat'*Y(N_var*(o-1)+N_gen+1:N_var*o);
                F(N_var*(s-1)+1:N_var*s) = F(N_var*(s-1)+1:N_var*s) - dt*a(s,o)*(P + [-D_gen.*Y(N_var*(o-1)+1:N_var*(o-1)+N_gen) - mat1*sin(dtheta); Y(N_var*(o-1)+1:N_var*(o-1)+N_gen); -mat2*sin(dtheta)]);
            end
        end
        F2 = -M2*F;
        for s = 1:Ns
            dW(N_var*(s-1)+1:N_var*s) = (lambda(s)*M - dt*J) \ F2(N_var*(s-1)+1:N_var*s);
        end
        W = W + dW;
        error = max(abs(M1*dW));
        Y = M1*W + repmat(y0,Ns,1);
        iter = iter + 1;
	end
    if(iter == maxiter)
        disp(['Max iteration reached, error ' num2str(error)])
    end
    y = Y((Ns-1)*N_var+1:end); 
end
