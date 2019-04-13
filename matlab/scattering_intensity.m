

function QI_matrix = scattering_intensity(mypath,startsnp,deltat,finalsnp)
%          mypath='../../Pbc_Ewald/scattering-pattern/output/timesteps/'
%          deltat = snapshots gap from scattering results
%          scattering_intensity(mypath,starttimestep,deltat,finaltimestep)
fsum = 0 ;
qpoints = '/q_points' ;
tsteps = '/timesteps/' ;
for ts = startsnp:deltat:finalsnp 
   intensity = load([mypath tsteps num2str(ts,'%08d')]);  
   fsum = fsum + intensity;
end


qpnts  = load([mypath qpoints ]);

kk = size(qpnts,1)


for i = 1:1:kk
qvect1 = (2.0 * pi * qpnts(i,1)) / 31.381;
qvect2 = (2.0 * pi * qpnts(i,2)) / 31.381;
qvect3 = (2.0 * pi * qpnts(i,3)) / 31.381;
mod_rvect = sqrt (qvect1 * qvect1 + qvect2 * qvect2 +qvect3 * qvect3); 
qpmag(i,1) = mod_rvect ;
end

QI_matrix = [qpnts qpmag fsum(:,2)];
