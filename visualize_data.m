%addpath <wherever you put the attached file>

% data dir
data_dir = '/Users/johnpeurifoy/Documents/Skewl/Thesis/PhysicsOfAI/ConsciousQuestion/';


% load data
rawdat = importdata([data_dir,'data/','snip_data.csv']);


Mtx = rawdat';

%% set the headers
chanList=1:104; % assuming there are 104 sensors
hdr.sfreq = 500; % sampling frq
hdr.nChans = length(chanList);
hdr.sensors.typestring= cell(1,length(chanList))
for i=1:length(chanList)
   hdr.sensors.typestring(i)={'lfp'}; 
end
hdr.tfirst =  0;
hdr.tlast = length(Mtx)/hdr.sfreq;

%% launch sig_browseraw
sig_browseraw(Mtx,'read_lfp','on','autoreject','off','submean','on','hdr',hdr)