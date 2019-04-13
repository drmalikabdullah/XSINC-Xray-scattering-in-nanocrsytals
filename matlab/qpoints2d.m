function final_data =qpoints2d(xs,xe,ys,ye,n,filename)
% xs = x-range starting point
% xe = x-range ending point
% ys = x-range starting point
% ye = x-range ensding point
% n = number of pints beetween starting and ending point
%filename = name of the file in which the qpoints will be saved



x = linspace(xs,xe,n);                                                                 
y = linspace(ys,ye,n);      
[X,Y] = meshgrid(x,y);
xx=reshape(X,prod(size(X)), 1) ;
yy=reshape(Y,prod(size(Y)), 1) ;
zz = zeros(size(xx,1),1);                                                                      
dens = [xx yy zz];
fid = fopen([filename],'w');
fprintf(fid,'% d % d % d\n',transpose(dens));

final_data = dens ;
