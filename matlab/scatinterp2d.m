
function  scatinterp2dd = scatinterp2d(mypath,startsnp,deltat,finalsnp,mfactor)
% example :    scatinterp(mypath,startsnp,deltat,finalsnp,mfactor)
%              mypath = /XSINC.1.03/3d-results/output
%              startsnp = 0
%              deltat   = 1000 or whatever you need
%              finalsnp = 40000 for 40 fs pulse
%              mfactor = a factor by which amount you want the refined interpolation
%              filename 'output data'

fsum = 0 ;
qpoints = '/q_points';
tsteps = '/timesteps/';
formfactor = '/formfactor/';
i = 0 ;
formfac = 0;
flu_info = '/flunce_info';
fluence_info = 0;

fluence_info = load([mypath flu_info]);
size(fluence_info,1);
for ts = startsnp:deltat:finalsnp
        intensity = load([mypath tsteps num2str(ts,'%08d')]);
            fsum = fsum + intensity;
            end 

            qpnts  = load([mypath qpoints ]);
            matrix = [qpnts fsum(:,5)];


scatinterp2dd.matrix = matrix ; 
