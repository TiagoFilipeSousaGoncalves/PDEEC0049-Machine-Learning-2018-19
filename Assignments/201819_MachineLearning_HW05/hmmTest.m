clear all;
close all;
%xx = '1245526462146146136136661664661636616366163616515615115146123562344';
xx = '166562663611111122';
% xx = '1215621624';
xx=xx-'0';% transform from char to double

AA = [0.95 0.05; 0.05 0.95]; %transition matrix
pi0 = [0.5; 0.5]; %initial probs
%NOTE: to have the same results as matlab functions (see the end of the script,use 
%pi0= AA(1,:)';

Px = [1/6 1/6 1/6 1/6 1/6 1/6; 0.1 0.1 0.1 0.1 0.1 0.5]; % emission probs
T = length(xx);
K = 2; % number of states

%% forward method - slide 31
alpha=pi0.*Px(:,xx(1));
alpha

allAlpha=alpha; % only needed for slide 40
for t=2:T
    alpha=Px(:,xx(t)).*(AA'*alpha); 
    allAlpha = [allAlpha alpha]; % only needed for slide 40
end

allAlpha
SequenceProbability=sum(alpha)
dddd

%% backward version  - slide 37
beta=ones(K,1);
allBeta = beta; % only needed for slide 40
for t=T-1:-1:1
    beta=AA*(Px(:,xx(t+1)).*beta);
    allBeta = [beta allBeta]; % only needed for slide 40
end
SequenceProbability=beta'*(pi0.*Px(:,xx(1)))
%code for slide 40
all=allBeta.*allAlpha

sum(all) %check: should equal SequenceProbability

stateProbabilities = all/SequenceProbability
dddd
%% viterbi - slide 53
Vtk=log(pi0) +  log(Px(:,xx(1)));
Ptr(:,t)=zeros(K,1);
logAA=log(AA);
for t=2:T
    [best, idx] = max(logAA + repmat(Vtk,1,K));
    best = best';
    idx = idx';
    Vtk=log(Px(:,xx(t))) + best; 
    Ptr(:,t)=idx;
end
[logprobstar, lastValue] = max(Vtk);
optimalSeq = lastValue;
for t=T:-1:2
    lastValue = Ptr(lastValue,t); 
    optimalSeq = [lastValue optimalSeq];
end
optimalSeq


%% results using MATLAB functions
PSTATES = hmmdecode(xx, AA, Px);
STATES = hmmviterbi(xx, AA, Px);
%check matlab agains our implementation
sum(STATES~=optimalSeq)
sum(sum(abs(PSTATES-stateProbabilities))) %minor differences (less than 1e-10) are expected in prob values due to numerical errors


%% supervised learning
trainingData = {'1245526462146146136136661664661636616366163616515615115146123562344';...
     '1665626636';...
     '43526163526661452631562533156243';...
     '63562553142526235234215634152535362621512521612612'};
trainingStates = {'1111122222222222222222222222222222222222222211111111111111111111111';...
     '2222222222';...
     '11111111122222111111111111111111';...
     '11111111111111111111111111222222222222222222222222'};

%sanity check
if length(trainingData)~=length(trainingStates)
    error ('inconsistent training data');
end
SIX=6;%6 - NUMBER OF DIFFERENT OUTCOMES
Px = zeros(K,SIX); 
AA = zeros(K,K);
pi0= zeros(K,1);
for i=1:length(trainingData)
    xx=trainingData{i}-'0';
    st=trainingStates{i}-'0';
    %sanity check
    if length(xx)~=length(st)
        error ('inconsistent training data');
    end
    %learn initial probs
    pi0(st(1))=pi0(st(1))+1;
    %learn emission probs
    ust=sort(unique(st));
    for j=1:length(ust)
        idx=find(st==ust(j));
        data=xx(idx);
        nn=hist(data,1:SIX);
        Px(j,:)=Px(j,:)+nn;
    end
    %learn transition matrix
    for j=1:length(st)-1
        AA(st(j),st(j+1))=1+AA(st(j),st(j+1));
    end
end   
Px=Px./repmat(sum(Px, 2),1,SIX)
AA=AA./repmat(sum(AA, 2),1,K)
pi0=pi0/sum(pi0)

dsadsa

%% unsupervised learning
trainingData = {'1245526462146146136136661664661636616366163616515615115146123562344'-'0';...
     '1665626636'-'0';...
     '43526163526661452631562533156243'-'0';...
     '63562553142526235234215634152535362621512521612612'-'0'};
Px = [1/6 1/6 1/6 1/6 1/6 1/6;
      0.1 0.1 0.1 0.1 0.1 0.5];
%Px=ones(2,6)/6;
AA = [0.95 0.05; 0.05 0.95];
pi0= [0.5; 0.5];

for iter=1:100
    %E step
    allGammas={};
    allXi={};
    for i=1:length(trainingData)
        xx=trainingData{i};
        alpha=pi0.*Px(:,xx(1));
        allAlpha=alpha; 
        T=length(xx);
        for t=2:T
            alpha=Px(:,xx(t)).*(AA'*alpha); 
            allAlpha = [allAlpha alpha]; 
        end    
        beta=ones(K,1);
        allBeta = beta; 
        for t=T-1:-1:1
            beta=AA*(Px(:,xx(t+1)).*beta);
            allBeta = [beta allBeta]; 
        end    
        all=allBeta.*allAlpha;
        ProbSeq=sum(all(:,1));
        gammas = all/ProbSeq;
        allGammas{i} = gammas;
        Xi={};
        for t=2:T
            aa=allAlpha(:,t-1);
            bb=allBeta(:,t);
            bb=bb.*Px(:,xx(t));
            Xi{t-1}=(aa*bb').*AA/ProbSeq;
        end
        allXi{i}=Xi;
    end
    %M step
    pi0= zeros(K,1);
    AA = zeros(K,K);
    Px = zeros(K,SIX);
    sgamma=zeros(K,1);
    for i=1:length(trainingData)
        gammas = allGammas{i};
        pi0 = pi0+gammas(:,1);
        Xi= allXi{i};
        for t=1:numel(Xi)
            AA=AA+Xi{t};
        end
        xx=trainingData{i};
        xx=repmat(xx,K,1);
        sgamma=sgamma+sum(gammas,2);
        for o=1:SIX
            idx = (xx==o);
            Px(:,o)=Px(:,o)+sum((idx.*gammas),2);
        end
    end
    pi0= pi0/sum(pi0);
    AA=AA./(sum(AA,2)*ones(1,K));
    Px=Px./repmat(sgamma,1,SIX);
end
pi0
AA
Px


[ESTTR,ESTEMIT] = hmmtrain(trainingData,[.95 0.05; 0.05 0.95], ones(2,6)/6)

