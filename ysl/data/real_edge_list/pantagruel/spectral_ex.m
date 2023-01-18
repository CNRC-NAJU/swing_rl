% In this example, a spectral decomposition of the network Laplacian 
% is carried out. The eigenvectors associated with some of the smallest
% non-zero eigenvalues are displayed.
%
%   Author: Laurent Pagnier (laurent.vincent.pagnier@gmail.com)
%   December 14, 2019
%
% Requirements: Matlab (R2015b or later) and Matpower (5.0 or later). 

function spectral_ex()
	clc
    close all

    pantagruel = pantagruel_case;
    N_bus = length(pantagruel.bus);
    N_line = length(pantagruel.branch);
	N_plot = 5;
    
	adj_matrix = zeros(N_bus,N_bus);
	for i=1:N_line
		adj_matrix(pantagruel.branch(i,1), pantagruel.branch(i,2)) = 1;
		adj_matrix(pantagruel.branch(i,2), pantagruel.branch(i,1)) = 1;
	end

	g = graph(adj_matrix, pantagruel.bus_name);
	incidence_mat = sparse([pantagruel.branch(:,1);pantagruel.branch(:,2)], [1:N_line 1:N_line], [ones(N_line,1);-ones(N_line,1)]);
    line_susceptance = 1./pantagruel.branch(:,4);
    
    disp('Diagonalizing the network Laplacian matrix...')
    Laplacian = incidence_mat * sparse(1:N_line, 1:N_line, line_susceptance) * incidence_mat';
    [v,d] = eig(full(Laplacian));
    d  = diag(d);
    
    % plotting eigenvalues and the eigenvectors corresponding to the
    % smallest non-zero eigenvalues
    figure
    plot(d)
    xlabel('mode index');
    ylabel('eigenvalue');
    
    for i=2:N_plot+1
        plot_results(pantagruel, g, v(:,i))
        colorbar('location','south');
        annotation('textbox',[0.4107 0.8428 0.2071 0.0470], 'String', ['eigenvector #' num2str(i)] ,'LineStyle','none');
    end
end

function plot_results(p, g, v)
	figure
	h = plot(g, 'XData', p.bus_coord(:,1), 'YData', p.bus_coord(:,2), 'EdgeColor', [.8 .8 .8], 'NodeColor',[1 0 0]);
	h.MarkerSize = 2;
    % define a color palette 
	cm = hsv;
    cm = [cm(43:-1:1,:); cm(end:-1:55,:)];
	colormap(cm)
	h.NodeCData = v;
	axis([-11 31 35 58])
	axis equal
	set(gca,'visible','off')  
end
