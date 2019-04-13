function all_qp = qpoints( sel_qp , n_grid , w)
%   This will Generate the Qpoints and will also check
%   that those qpoints are inside the Ewald's Spehere
%    example:
%           qpoints( [1 1 1 : 2 2 2 ...] , number of grid points , limit )
%    A is Angstrom

%sel_qp = [1 1 1]
sel_qp = [1 1 1]
%sel_qp = [1 0 1;1 0 2 ;1 0 3; 2 0 1;2 0 2;2 0 3]
n_qp = size( sel_qp , 1 ) ;
n_grid = 1000;
w = 0.05 ;
range = linspace( -w , w , n_grid) ;
 %all_qp = zeros( n_qp * n_grid * 3 , 3 ) ;
all_qp = [] ;
for ii = 1 : n_qp
    for jj = 1 : 3
      %all_qp( (ii-1)*n_qp+1:ii*n_qp  ,jj) = linspace( sel_qp(jj)-w , sel_qp(jj)+w , n_grid) ; 
      x = zeros(n_grid,3) ; x(:,jj) = range ;
      all_qp = [ all_qp ; repmat(sel_qp(ii,:),n_grid,1) + x ] ;
    end
end

return 
 ewald_sphere = (2 * 2 * pi) / lambda ; 
 cc = 1 ;  


for ri = -n : n                                    
    for rj = 3 : 3                                           
         for rk = 3 : 3                                           
             test1 = ( 2.0 * pi * ri ) / unitcell_size ;         
             test2 = ( 2.0 * pi * rj ) / unitcell_size ;    
             test3 = ( 2.0 * pi * rk ) / unitcell_size ;       
            mod_rvect = sqrt (test1 * test1 + test2 * test2 + test3 * test3);
           if mod_rvect <= ewald_sphere    
          % qvect(cc,1) = ( 2.0 * pi * ri ) / unitcell_size ;     
          % qvect(cc,2) = ( 2.0 * pi * rj ) / unitcell_size ;  
          % qvect(cc,3) = ( 2.0 * pi * rk ) / unitcell_size ;  
            qvect(cc,1) = ri ;
            qvect(cc,2) = rj ;   
            qvect(cc,3) = rk ;   
            cc = cc + 1 ;       
            end
         end  
     end 
 end                                                                 
 fid = fopen ('input_qpoints.txt','w');
 fprintf(fid,'%d %d %d\n',transpose(all_qp));                 
