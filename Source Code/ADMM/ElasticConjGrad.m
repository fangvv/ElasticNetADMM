function [w,v,W_i, V_i, history] = ElasticConjGrad(dataAll, lambda, mu,rho, patientNo)

t_start = tic;%设置时间戳
QUIET    = 0;
ABSTOL   = 1e-4;%定义绝对容忍度和相对容忍度
%ABSTOL   = 1e-6;
RELTOL   = 1e-2;
%RELTOL   = 1e-4;

%Data preprocessing
[examples,Col] = size(dataAll);

% [examples, feature] = size(dataAll);
trainingNum = floor(examples*0.7);
testExamples = dataAll(trainingNum+1:examples,:);
dataAll = dataAll(1:trainingNum,:);

data = dataAll(:, 1:Col-1); 
b0 = dataAll(:, Col);
features = Col-1;
onepatientSample = floor(trainingNum/patientNo);% 每个病人提供样本数

MAX_ITER = 10000;
%初始值赋值
flag = 0;
w = zeros(1,features);        % 特征向量拟合系数
v = 0;
W_i = zeros(patientNo,features);        % 辅助变量
V_i = zeros(patientNo,1);

gama_w = zeros(patientNo,features);        % 辅助变量平均值
gama_v = zeros(patientNo,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k=1:MAX_ITER
    wold = w;
    vold = v;
    
    % w-update
    a = ((mean(W_i,1) + mean(gama_w,1)/rho)*rho*patientNo)/(rho*patientNo+mu);  %mean(a,1) :纵向求均值
    kappa = lambda/(rho*patientNo+mu);        
    w = shrinkage(a, kappa);               %[x - kappa]+  - [-x - kappa]+
    v = mean(V_i,1) + mean(gama_v,1)/rho;
    
    W_iold = W_i;
    V_iold = V_i;
 
    % 使用共轭梯度法更新 W_i-update  V_i-update
    for i = 1:patientNo
        A_i = data(onepatientSample*(i-1)+1:onepatientSample*i,:);
        b_i = b0(onepatientSample*(i-1)+1:onepatientSample*i,1);
        C = A_i;
        B = b_i - V_i(i,1);
        W_col = w;
        TOLERANCE = 1e-5;
        ITER = 50;
        I = eye(features);
        x = zeros(1,features);
        f = @(w) (1/(2)*sum((C*w' - B).^2) + (rho/2)*(w*(w-2*W_col + 2/rho*gama_w(i,:))') + (rho/2)*V_i(i,1)*(V_i(i,1) - 2*v + 2/rho *gama_v(i,1)) );
        
        fx = f(x);
        g = (x*C'*C - B'*C) + rho*x - rho*W_col + gama_w(i,:);
        pk = -g;
        if norm(g,2) > TOLERANCE
           for iter = 1:ITER
                lambdak = 0.01;
                for t = -1:2:1
                     if f(x + lambdak*pk) > f(x + t*pk)
                        lambdak = t;
                    end;
                end;
                xold = x;
                gold = g;
                x = x + lambdak*pk;
                g = (x*C'*C - B'*C) + rho*x - rho*W_col + gama_w(i,:);
                betak = (g*(g-gold)')/norm(gold,2)^2;
                betak=(g*(g-gold)')/(pk*(g-gold)');
                pk = -g + betak*pk;
                if norm(g,2) < TOLERANCE
                    break;
                end
            end
        end
        W_i(i,:) = x;
    end;
    for i = 1:patientNo
        A_i = data(onepatientSample*(i-1)+1:onepatientSample*i,:);
        b_i = b0(onepatientSample*(i-1)+1:onepatientSample*i,1);
        C = A_i*W_i(i,:)'-b_i;
        W_col = w;
        V_col = v;
        wi = W_i(i,:); 
        TOLERANCE = 1e-5;  
        ITER = 50;
        x = 0;
        f = @(v) ( 1/(2)*sum((C + v).^2) + (rho/2)*((wi*(wi-2*W_col + 2/rho*gama_w(i,:))')) + (rho/2)*v*(v - 2*V_col + 2/rho *gama_v(i,1)) );

        fx = f(x);
        g = (x*onepatientSample + sum(C)) + rho*x - rho*V_col + gama_v(i,1);
        pk = -g;
        if norm(g,2) > TOLERANCE
           for iter = 1:ITER
                lambdak = 0.01;
                for t = -1:2:1
                    if f(x + lambdak*pk) > f(x + t*pk)
                        lambdak = t;
                        
                    end;
                end;
                xold = x;
                gold = g;
                x = x + lambdak*pk;
                g = (x*onepatientSample + sum(C)) + rho*x - rho*V_col + gama_v(i,1);
                betak = (g*(g-gold)')/norm(gold,2)^2;
                
              
                
                pk = -g + betak*pk;

                if norm(g,2) < TOLERANCE
                    break;
                end
            end
        end
        V_i(i,:) = x;
        
    end;
    
    % gama_w gama_v-update
    for i = 1:patientNo
        gama_w(i,:) = gama_w(i,:) + rho*(W_i(i,:) - w);
        gama_v(i) = gama_v(i) + rho*(V_i(i) - v);
    end;
            
    dualObjective = 0;           % 计算对偶损失函数
    primObjective = 0;           % 计算原始损失函数
    for i=1:patientNo
        A_i = data(onepatientSample*(i-1)+1:onepatientSample*i,:);
        b_i = b0(onepatientSample*(i-1)+1:onepatientSample*i,1);
        dualObjective = dualObjective + 1/(2)*sum((A_i*W_i(i,:)' + V_i(i) - b_i).^2) + (W_i(i,:) - w)*gama_w(i,:)'+gama_v(i)*(V_i(i)-v) + 0.5*rho*((W_i(i,:) - w)*(W_i(i,:) - w)'+(V_i(i)-v)^2);
        primObjective = primObjective + 1/(2)*sum((A_i*W_i(i,:)' + V_i(i) - b_i).^2);
    end;
     history.dualObjective(k) = dualObjective + (lambda*norm(w,1)+0.5*mu*norm(w,2));
    history.primObjective(k) = primObjective + (lambda*norm(w,1)+0.5*mu*norm(w,2));
    %基于Elastic Net的目标函数值
     history.beforeADMMObjective(k) = 1/2*sum((data*w' + v - b0).^2) + lambda*norm(w,1)+0.5*mu*norm(w,2);
    %% 计算误差：训练集
    %计算R方值
    fenzi = sum( (b0 - data*w'- v).^2 );
    fenzi_MAE=sum(abs(b0-data*w'-v));
    fenzi_MAPE=sum(abs((b0 - data*w'- v)./b0));
    fenmu = sum( (b0 - mean(b0)).^2 );
    history.R2(k) = 1 - fenzi/fenmu;

    %计算R方校正决定系数
    history.R2_adjusted(k) = 1 - ((1-history.R2(k))*(trainingNum-1))/(trainingNum - features - 1);
    
    %ADMM训练集误差
    history.MSE(k) = (fenzi/trainingNum);
    history.MAE(k)=(fenzi_MAE/trainingNum);
    history.RMSE(k)=sqrt(fenzi/trainingNum);
    history.MAPE(k)=(fenzi_MAPE/trainingNum);
    
    
    %%自己拟合的系数差别较大的数据：特征个数是10:使用前向选择法
    optW = [ 0.0200   -0.0600   22.0400  -10.0500    2.0300    7.9800   -9.0300   20.0400    0.0500    0.0100];
    optV = [ 0];
    %%  利用Python实现的集中式ElasticNet最优解
    %optW = [ 5.08904890e-02 -1.88009150e-01  2.20422902e+01 -1.00623581e+01 2.02747251e+00  8.02660560e+00 -9.06965189e+00  2.00206366e+01 -8.84298299e-02 -1.71624955e-02];
    %optV = [ -0.00110583];

    
    fenzi1 = sum( (b0 - data*optW'- optV).^2 );
    fenzi1_MAE=sum(abs(b0-data*optW'-optV));
    fenzi1_MAPE=sum(abs((b0 - data*optW'- optV)./b0));
    %集中式结果的训练集误差
    history.centralizeMSE(k) = (fenzi1/trainingNum);
    history.centralizeMAE(k)=(fenzi1_MAE/trainingNum);
    history.centralizeRMSE(k)=sqrt(fenzi1/trainingNum);
   history.centralizeMAPE(k)=(fenzi1_MAPE/trainingNum);
    
    matrixPrim = zeros(1,features+1);     % 计算的[w,v]值
    matrixPrim(1,1:features) = w;
    matrixPrim(1,features+1) = v;
    
    matrixDual = zeros(1,features+1);     % 计算的[W_i,V_i]值
    matrixDual(1,1:features) = mean(W_i,1);
    matrixDual(1,features+1) = mean(V_i,1);
    
    matrixDualOld = zeros(1,features+1);     % 计算的[W_i,V_i] old 值
    matrixDualOld(1,1:features) = mean(W_iold,1);
    matrixDualOld(1,features+1) = mean(V_iold,1);
    
    matrixGama = zeros(1,features+1);     % 计算的[W_i,V_i]值
    matrixGama(1,1:features) = mean(gama_w,1);
    matrixGama(1,features+1) = mean(gama_v,1);
    
    history.r_norm(k)  = norm(matrixPrim - matrixDual);     % 计算原问题残差
    history.s_norm(k)  = norm(-rho*(matrixDual - matrixDualOld));     % 对偶问题相对残差
    
    history.eps_pri(k) = sqrt(features+1)*ABSTOL + RELTOL*max(norm(matrixPrim), norm(-matrixDual));     % 原始残差容忍度
    history.eps_dual(k)= sqrt(features+1)*ABSTOL + RELTOL*norm(rho*matrixGama);
    
%     fprintf(fobj,'%-5d %-5f %-5f %-5f %-5f %-5f %-5f\n',k,primObjective,dualObjective,history.r_norm(k),history.s_norm(k),history.eps_pri(k),history.eps_dual(k)); % 写入文件（k,原函数值,对偶函数值，原问题残差，对偶残差，原容忍度，对偶容忍度）

    
    disp( [ '   iter     =  ', sprintf( '%d ', k ) ] );
    disp( [ '   w     = [ ', sprintf( '%7.4f ', w ), ']' ] );
    disp( [ '   v     = [ ', sprintf( '%7.4f ', v ), ']' ] );
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.dualObjective(k));
    end
    

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         history.OriginalResidualsIter = k;
         flag = 1;
         break;
    end 
    
end;

history.optiobjval = 1/(2)*sum((data*optW' + optV - b0).^2) + lambda*norm(optW,1)+0.5*mu*norm(optW,2);
history.optW = optW;
history.optV = optV;
%% 计算误差：测试集
[testM, testN] = size(testExamples);
data1 = testExamples(:, 1:testN-1);
b1 = testExamples(:, testN);
 %计算R方值
fenziTest = sum( (b1 - data1*w'- v).^2 );
fenziTest1= sum(abs(b1-data1*w'-v));
fenziTest2= sum(abs((b1 - data1*w'- v)./b1));
fenmuTest = sum( (b1 - mean(b1)).^2 );
history.testR2 = 1 - fenziTest/fenmuTest;

%计算R方校正决定系数
history.testR2_adjusted = 1 - ((1-history.R2(k))*(testM-1))/(testM - features - 1);

%训练集误差
history.testMSE = (fenziTest/testM);
history.testMAE=(fenziTest1/testM);
history.testRMSE=sqrt(fenziTest/testM);
history.testMAPE= (fenziTest2/testM);

if ~QUIET
    toc(t_start);
end
end
function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end