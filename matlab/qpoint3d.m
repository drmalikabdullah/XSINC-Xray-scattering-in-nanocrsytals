
function  qpointoutput = qpoint3d(xs,dltax,xe,ys,dltay,ye,zs,dltaz,ze,filename)
% example :    qpoint3d(xs ,dltax ,xe ,ys ,dltay ,ye ,zs ,dltaz ,ze,filename)
%              xs = starting qpoint along qx direction
%              dltax = delta qx
%              xe = ending qpoint along qx direction
%              Same goes with the other
%              filename = specify the filename



[X,Y,Z]= meshgrid( [xs:dltax:xe] , [ys:dltay:ye]  , [zs:dltaz:ze]  );
xx=reshape(X,prod(size(X)), 1) ;
yy=reshape(Y,prod(size(Y)), 1) ;
zz=reshape(Z,prod(size(Z)), 1) ;
dens = [xx yy zz];
size(dens)

fid = fopen([filename],'w');
fprintf(fid,'% d % d % d\n',transpose(dens));
qpointoutput.x = X ;
qpointoutput.y = Y ;
qpointoutput.z = Z ;
fclose('all');
