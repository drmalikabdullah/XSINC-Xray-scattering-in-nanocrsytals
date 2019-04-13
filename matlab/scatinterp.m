
function  scatinterp3d = scatinterp(mypath,startsnp,deltat,finalsnp,mfactor)
% example :    scatinterp(mypath,startsnp,deltat,finalsnp,mfactor)
%              mypath = /XSINC.1.03/3d-results/output
%              startsnp = 0
%              deltat   = 1000 or whatever you need
%              finalsnp = 40000 for 40 fs pulse
%              mfactor = a factor by which amount you want the refined interpolation

fsum = 0 ;
qpoints = '/q_points' ;
tsteps = '/timesteps/' ;

for ts = startsnp:deltat:finalsnp
    intensity = load([mypath tsteps num2str(ts,'%08d')]);
    fsum = fsum + intensity;
end

qpnts  = load([mypath qpoints ]);
matrix = [qpnts fsum(:,5)];
matrix_nof0 = [qpnts fsum(:,2)];


plrangex = (max(qpnts(:,1)) + min(qpnts(:,1))) /2; 
plrangey = (max(qpnts(:,2)) + min(qpnts(:,2))) /2;
plrangez = (max(qpnts(:,3)) + min(qpnts(:,3))) /2; 


mtrixs=(size(qpnts,1))^(1/3);
mtrixs=round(mtrixs); 
qvect_a_s = qpnts(1,1);
qvect_b_s = qpnts(1,2);
qvect_c_s = qpnts(1,3);

qvect_a_e = qpnts(end,1);
qvect_b_e = qpnts(end,2);
qvect_c_e = qpnts(end,3);

xx =0 ; yy = 0 ; zz = 0;
xx = unique(qpnts(:,1));
yy = unique(qpnts(:,2));
zz = unique(qpnts(:,3));


qvect_diff_a = (xx(2,1) - xx(1,1)) * 10000;
qvect_diff_b = (yy(2,1) - yy(1,1)) * 10000;
qvect_diff_c = (zz(2,1) - zz(1,1)) * 10000;

qvect_diff_a=round(qvect_diff_a)/10000;
qvect_diff_b=round(qvect_diff_b)/10000;
qvect_diff_c=round(qvect_diff_c)/10000;

new_qvect_diff_a = qvect_diff_a / mfactor ;
new_qvect_diff_b = qvect_diff_b / mfactor ;
new_qvect_diff_c = qvect_diff_c / mfactor ;

x = reshape(matrix(:,4),mtrixs,mtrixs,mtrixs);
xi = reshape(matrix_nof0(:,4),mtrixs,mtrixs,mtrixs);



%[X,Y,Z]=    meshgrid( qvect_a_s:0.0012:qvect_a_e , qvect_b_s:0.0012:qvect_b_e  , qvect_c_s:0.002:qvect_c_e  );
[X,Y,Z]=    meshgrid( qvect_a_s:qvect_diff_a :qvect_a_e , qvect_b_s:qvect_diff_b :qvect_b_e  , qvect_c_s:qvect_diff_c:qvect_c_e  );
[Xq,Yq,Zq]= meshgrid( qvect_a_s:new_qvect_diff_a:qvect_a_e , qvect_b_s:new_qvect_diff_b:qvect_b_e  , qvect_c_s:new_qvect_diff_c:qvect_c_e  );
Vq = interp3(X,Y,Z,x,Xq,Yq,Zq,'spline');

figure ;
slice(Xq,Yq,Zq,Vq,[plrangex plrangex],plrangey,plrangez);
title 'Interpolated-No-f0'

shading flat

figure;
VVq = interp3(X,Y,Z,xi,Xq,Yq,Zq,'spline');
slice(Xq,Yq,Zq,VVq,[plrangex plrangex],plrangey,plrangez);
title 'Interpolated'
shading flat ;
scatinterp3d.X = [Xq];
scatinterp3d.Y = [Yq];
scatinterp3d.Z = [Zq];
scatinterp3d.Inof0 = [Vq];
scatinterp3d.I = [VVq];
%v2 = gdsdf ;

scatinterp3d.Eff_f0 =sqrt( sum(scatinterp3d.I(:)) / sum(scatinterp3d.Inof0(:))) ;
scatinterp3d.sumI = sum(scatinterp3d.I(:)) ;
scatinterp3d.sumInof0 = sum(scatinterp3d.Inof0(:)) ;
%varargout = { scatinterp3d , v2 } ; 
