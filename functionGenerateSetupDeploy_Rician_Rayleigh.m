function [R_AP_Rayleigh,R_AP_Rician,HMean_Withoutphase] = functionGenerateSetupDeploy_Rician_Rayleigh(M,K,N,nbrOfSetups,correlatedShadowing,probLoS_Rician,probLoS_Rayleigh)
%%=============================================================
%The file is used to generate the system set up over both the Rician and Rayleigh fading channels of the paper:
%
%Zhe Wang, Jiayi Zhang, Hao Lei, Dusit Niyato, and Bo Ai, "Optimal Bilinear Equalizer Beamforming Design for Cell-Free Massive MIMO Networks with Arbitrary Channel Estimators,"
%IEEE Transactions on Vehicular Technology, to appear, 2024, %doi: 10.1109/TVT.2024.3520500.
%
%Download article: https://arxiv.org/abs/2503.00763 or https://ieeexplore.ieee.org/document/10810748
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%paper as described above.
%%=============================================================

%% Define simulation setup

%Size of the coverage area (as a square with wrap-around)
cellRange = 1000; %meter

%Communication bandwidth
B = 20e6;

%Noise figure (in dB)
noiseFigure = 7;

%Compute noise power
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;


%Pathloss parameters
alpha_LoS = 26;
constantTerm_LoS = 30.18;

alpha_NLoS = 38;
constantTerm_NLoS = 34.53;


%Standard deviation of the shadow fading
sigma_sf = 8;

%Decorrelation distance of the shadow fading
decorr = 100;

%Shadow fading parameter
delta = 0.5;

%Height difference between an AP and a UE
APheigth = 12.5;  
UEheigth = 1.5;  
VerticalDistance = APheigth-UEheigth;

%The minimum allowed distance (Access Point to User Equipment)
dmin = 10; 

%Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2; %Half wavelength distance

%Angular standard deviation around the nominal angle (measured in degrees)
ASDdeg = 30;

%Dropping all UEs while minimum distance requirement is satisfied.
droppedUEs=0;

%Deploy APs randomly
APpositions = cellRange*(rand(M,1) + 1i*rand(M,1));

%Prepare to save results
R_AP_Rayleigh = zeros(N,N,M,K,nbrOfSetups);
R_AP_Rician = zeros(N,N,M,K,nbrOfSetups);
channelGain_LoS_Rician = zeros(M,K,nbrOfSetups);
channelGain_NLoS_Rician = zeros(M,K,nbrOfSetups);
channelGain_NLoS_Rayleigh = zeros(M,K,nbrOfSetups);
HMean_Withoutphase = zeros(M*N,K,nbrOfSetups);

%% Go through all setups
for n = 1:nbrOfSetups
 
%------Depoly UEs randomly

%--Compute alternative AP locations by using wrap around
wrapHorizontal = repmat([-cellRange 0 cellRange],[3 1]);
wrapVertical = wrapHorizontal';
wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
APpositionsWrapped = repmat(APpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[M 1]); 

%--Prepare to save the distances and UE positions
distanceAPtoUE = zeros(M,K);
distanceAPtoAP = zeros(M,M); 
distanceUEtoUE = zeros(K,K); 
UEpositions = zeros(K,1);

%--Dropping all UEs
while droppedUEs <K
    
    UEposition = cellRange*(rand(1,1) + 1i*rand(1,1));
    HorizontalDistance = abs(APpositions-UEposition);
    distance = sqrt(VerticalDistance.^2 + HorizontalDistance.^2);
    
    if isempty(distance(distance<dmin))
        droppedUEs = droppedUEs+1;
        distanceAPtoUE(:,droppedUEs)=distance;
       
        %Store UE positions
        UEpositions(droppedUEs)=UEposition;
    end
end


%---Calculate the distances between all AP and UE pairs

%Distances between APs
for m=1:M
    distanceAPtoAP(:,m) = abs(APpositions-APpositions(m));
end

%Distances between UEs
 for k=1:K
    distanceUEtoUE(:,k) = abs(UEpositions-UEpositions(k));
 end
 
%-----Calculate Channel Coefficients

%Create covarince functions for each pair
covMatrixAP = 2.^(-distanceAPtoAP/decorr);
covMatrixUE = 2.^(-distanceUEtoUE/decorr);

%Create shadow fading realizations
shadowFadingAP = sqrtm(covMatrixAP)*randn(M,1); 
shadowFadingUE = sqrt(covMatrixUE)*randn(K,1);

%Create the resulting shadow fading matrix
 shadowFadingMatrix=zeros(M,K);
 for k=1:K
     for m=1:M
        shadowFadingMatrix(m,k)=sqrt(delta)*shadowFadingAP(m) +sqrt(1-delta)*shadowFadingUE(k);
     end
 end
 
 %Scale with variance 
 if correlatedShadowing == 1
    %Correlated Shadow Fading Matrix in dB
    shadowFading = sigma_sf*shadowFadingMatrix; 
 else
    %Uncorrelated Shadow Fading Matrix in dB
    shadowFading = sigma_sf*randn(M,K); 
 end

 
%Prepare to save the result
RicianFactor = zeros(M,K);
channelGaindB_Rician = zeros(M,K);
channelGain_Rician = zeros(M,K);

channelGaindB_Rayleigh = zeros(M,K);
channelGain_Rayleigh = zeros(M,K);
% HMean_SingalAntenna = zeros(M,K);
%Go through all UEs
 for k = 1:K
     
     [distances_Hori,whichpos] = min(abs(APpositionsWrapped - repmat(UEpositions(k),size(APpositionsWrapped))),[],2);
     distances = sqrt(VerticalDistance^2+distances_Hori.^2);
%      distances = distances_Hori;
     
     %Path-loss calculation "Cost 231 Walfish-Ikegami Model"
     %before adding shadow fading
     betaLoS = constantTerm_LoS + alpha_LoS*log10(distances);
     betaNLoS = constantTerm_NLoS + alpha_NLoS*log10(distances);
     
%      probLoS = ones(size(distances));
     
     %Calculate the distance based Rician Factor 
     RicianFactor(:,k) = 10.^(1.3-0.003*distances);
     
     %Save the channel gains (in this setup each pair has a LoS path)
     channelGaindB_Rician(probLoS_Rician==1,k)=-betaLoS(probLoS_Rician==1);
     channelGaindB_Rayleigh(probLoS_Rayleigh==0,k)=-betaNLoS(probLoS_Rayleigh==0);

     channelGaindB_Rician(:,k) = channelGaindB_Rician(:,k)+shadowFading(:,k)-noiseVariancedBm+30;
     channelGain_Rician(:,k) = db2pow(channelGaindB_Rician(:,k));
     
     channelGaindB_Rayleigh(:,k) = channelGaindB_Rayleigh(:,k)+shadowFading(:,k)-noiseVariancedBm+30;
     channelGain_Rayleigh(:,k) = db2pow(channelGaindB_Rayleigh(:,k));

     
     %Scale with Rician factor
     channelGain_LoS_Rician(probLoS_Rician==1,k,n) = sqrt(RicianFactor(probLoS_Rician==1,k)./(RicianFactor(probLoS_Rician==1,k) +1 )).*sqrt(channelGain_Rician(probLoS_Rician==1,k));
     channelGain_NLoS_Rician(probLoS_Rician==1,k,n) = (1./(RicianFactor(probLoS_Rician==1,k) +1 )).*(channelGain_Rician(probLoS_Rician==1,k));
     channelGain_NLoS_Rayleigh(probLoS_Rayleigh==0,k,n) = channelGain_Rayleigh(probLoS_Rayleigh==0,k); %note that probLoS is always one in the manuscript


      
%      %Go through all APs
     for m = 1:M
            
            %Compute nominal angle between UE k and AP m
            angletoUE = angle(UEpositions(k)-APpositionsWrapped(m,whichpos(m)));
            
            %Generate normalized spatial correlation matrix using the local
            %scattering model
            R_AP_Rician(:,:,m,k,n) = channelGain_NLoS_Rician(m,k,n)*functionRlocalscattering(N,angletoUE,ASDdeg,antennaSpacing);
            R_AP_Rayleigh(:,:,m,k,n) = channelGain_NLoS_Rayleigh(m,k,n)*functionRlocalscattering(N,angletoUE,ASDdeg,antennaSpacing);

            %Generate HMean_Withoutphase in Multi-Antenna case
            HMean_Withoutphase((m-1)*N+1:m*N,k,n) = channelGain_LoS_Rician(m,k)*(exp(1i*2*pi.*(0:(N-1))*sin(angletoUE)*antennaSpacing));

            
     end
     


 end
 
end


 
 