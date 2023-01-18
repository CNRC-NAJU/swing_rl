%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     PanTaGruEl - a pan-European transmission grid and electricity generation model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

When publishing results based on this data set, please cite:

L. Pagnier, P. Jacquod , Inertia location and slow network modes determine 
disturbance propagation in large-scale power grids. PLOS ONE 14(3): e0213550, 2019.
https://doi.org/10.1371/journal.pone.0213550

and

M. Tyloo, L. Pagnier, P. Jacquod, The key player problem in Ccomplex oscillator
networks and electric power grids: Resistance centralities identify local
vulnerabilities, Science Advances 5(11): eaaw8359, 2019.
https://doi.org/10.1126/sciadv.aaw8359

Contact:
  Laurent Pagnier, laurent.vincent.pagnier@gmail.com

December 14, 2019

Licensed under the Creative Commons Attribution 4.0 International license,
http://creativecommons.org/licenses/by/4.0/


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		 Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This csv version is provided with limited support. If you are interested
in a port to a particular programming language, please contact us.

********************************
        nomenclature
********************************

Most column headers are self-explaining. The definitions of primary control, 
inertia and frequency load coefficient, as well as information on how the model
was built, are found on https://doi.org/10.1371/journal.pone.0213550.s002

Population prop(ortion) provides for each bus the share of the national
population that is supplied through this particular bus.
 
Generator technologies:
  CG: Gas
  DA: Dam
  FM: Mixed fossil
  FO: Fossil oil
  HC: Hard coal
  HY: Mixed hydro
  LI: Lignite
  NU: Nuclear
  OT: Other
  PS: Pumped-storage hydro
  RR: Run-of-the-river


********************************
      transformer model
********************************

Transformers are modelled as impedances in series with ideal transformers
converting the nominal voltage V1n to nominal voltage V2n.
			
       impedance    +    ideal transformer
      -----------          ---------
-----|  Z=R+iX   |--------| V1n/V2n |-----
      -----------          ---------
                       
     
     
     
     
     
     
