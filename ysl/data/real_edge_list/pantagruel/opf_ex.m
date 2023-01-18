% In this example, optimal power flow computations are carried out. A
% first OPF is performed, then nodal demands are modified and a second OPF 
% is performed. Finally, the results are displayed using Matlab's graph.
%
%   Author: Laurent Pagnier (laurent.vincent.pagnier@gmail.com)
%   December 14, 2019
%
% Requirements: Matlab (R2015b or later) and Matpower (5.0 or later).  

function opf_ex()
    clc
    close all
    disp('Starting the simulation...')
	% load the model and perform an OPF
    pantagruel = pantagruel_case;
    pantagruel.gen(:,10) = 0; % no minimal output
    
	mpopt = mpoption('verbose',0,'out.all',0); % remove the OPF display
	mpopt.model = 'DC';
	pantagruel = runopf(pantagruel, mpopt); % run a DC opf with Matpower.
    plot_results(pantagruel);
    disp('1st OPF done.')
	
    % change nodal demands and perform another OPF
	% input: national loads in MW
    disp('Changing nodal demands...')
	national_load = {
		'AL', 700;
		'AT', 8300;
		'BA', 1400;
		'BE', 10200;
		'BG', 4100;
		'CH', 7000;
		'CZ', 7600;
		'DE', 62300;
		'DK', 3700;
		'ES', 26700;
		'FR', 59400;
		'GR', 6100;
		'HR', 1900;
		'HU', 2500;
		'IT', 33200;
		'LU', 600;
		'ME', 300;
		'MK', 800;
		'NL', 12300;
		'PL', 16200;
		'PT', 5300;
		'RO', 5500;
		'RS', 3100;
		'SI', 1500;
		'SK', 3200;
		};

	for i=1:length(national_load)
		id = strcmp(pantagruel.bus_country, national_load{i,1}); % find buses in ith country
		pantagruel.bus(id,3) = national_load{i,2} * pantagruel.bus_pop_prop(id); % distribute the national load
	end
	
	pantagruel = runopf(pantagruel, mpopt);
    plot_results(pantagruel);
    disp('2nd OPF done.')
    disp('Figures may take a few seconds to appear.')
end


function plot_results(p)
	N_bus = length(p.bus);
    N_line = length(p.branch);
    
	% plot the results with Matlab's graph handling
	adj_matrix = zeros(N_bus,N_bus);
	for i=1:N_line
        adj_matrix(p.branch(i,1), p.branch(i,2)) = 1;
        adj_matrix(p.branch(i,2), p.branch(i,1)) = 1;
	end
	g = graph(adj_matrix,p.bus_name);
    
	load = p.bus(:,3);
	gen = zeros(N_bus,1);
	id_gen = p.gen(:,1);
	gen(id_gen) = p.gen(:,2);

	Pij = zeros(N_bus, N_bus);
	Pij_max = zeros(N_bus, N_bus);
	for i=1:N_line
		id1 = max(p.branch(i,1), p.branch(i,2));
		id2 = min(p.branch(i,1), p.branch(i,2));
		Pij(id1,id2) = Pij(id1,id2) + abs(p.branch(i,14))+1E-16; % There are sometimes parallel lines, we sum their power flows.
		Pij_max(id1,id2) = Pij_max(id1,id2) + abs(p.branch(i,6));
	end
	powerflows = nonzeros(Pij);
	powerflows_max = nonzeros(Pij_max);
    
	figure
	subplot(2,2,1)
	h = plot(g, 'XData', p.bus_coord(:,1), 'YData', p.bus_coord(:,2), 'EdgeColor', [.8 .8 .8], 'NodeColor',[1 0 0]);
	h.MarkerSize = 0.1*sqrt(load)+1E-9;
    annotation('textbox',[0.2510 0.8904 0.0865 0.0476],'String','Load','LineStyle','none');
    
	subplot(2,2,2)
	h = plot(g, 'XData', p.bus_coord(:,1), 'YData', p.bus_coord(:,2), 'EdgeColor', [.8 .8 .8], 'NodeColor',[0 0 1]);
	h.MarkerSize = 0.1*sqrt(gen)+1E-9;
    annotation('textbox',[0.6599 0.8904 0.1508 0.0476],'String','Generation','LineStyle','none');

	subplot(2,2,3)
	h = plot(g, 'XData', p.bus_coord(:,1), 'YData', p.bus_coord(:,2), 'EdgeColor', [.8 .8 .8], 'NodeColor',[0 0 1]);
	h.MarkerSize = 1E-9;
	h.LineWidth = 1;
	h.EdgeAlpha = 1;
    % define a color palette 
    cm = hsv;
    cm = [cm(43:-1:1,:); cm(end:-1:55,:)];
	colormap(cm)	
	h.EdgeCData = powerflows./powerflows_max;
    annotation('textbox',[0.2521 0.4190 0.0997 0.0470],'String','Flows','LineStyle','none');
    
	for i=1:4
		subplot(2,2,i)
		axis([-11 31 35 58])
		axis equal
		set(gca,'visible','off')
	end
end
