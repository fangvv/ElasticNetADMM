function [ wResult, lossResult, history ] = stageWise( x, y, eps, runtime, testExamples)
%     t_start = tic;%����ʱ���
    [m,n] = size(x);%���ݼ��Ĵ�С
    wResult = zeros(runtime, n);%���յĽ��
    lossResult = zeros(runtime, 1);%��ʧ����
    lossResultOld = 10000000;
    w = zeros(n,1);
    yTest = x*w;
    wMax = zeros(n,1);
    eps = 1e-2;
    flag = 0;
    for i = 1:runtime
        ws = w'%���ÿһ�μ��������Ȩ��
        wold = w;
        
        lowestError = inf;%������Сֵ
        for j = 1:n
            for sign = -1:2:1
                wTest = w;%��ʼ��
                wTest(j) = wTest(j)+eps*sign;%ֻ�ı�һά����
                yTest = x*wTest;
                %�����
                rssE = rssError(y, yTest);
                if rssE < lowestError%����ã����滻
                    lowestError = rssE;
                    wMax = wTest;
                end
            end
        end
        w = wMax;
        wResult(i,:) = w;
        lambda = 1.0;
        %�������mu�������ֵΪ1.0
        mu=1;
        lossResult(i,1) = Objective(x, w, y, lambda,mu); 
        
        loss = lossResult(i,1)
%         flag2 = abs(lossResultOld - lossResult(i,1));
        MSEN = rssError(y,yTest)/m;
        history.MSEN(i) = MSEN;
        flag1 = MSEN;
        %% �жϵ�ǰ��wֵ�Ƿ�����
        if(flag1 <= 0.5 && flag == 0)
            disp('***********************');
            history.myIter = i;
            flag = 1;
%             toc(t_start);
            
        end
    end
    %% MSE�������
            history.MSE = rssError(y,yTest)/m;
            %% R2ֵ
            history.R2 = 1 - rssError(y,yTest)/sum((y - mean(y)).^2);
            %% R2Adjusted
            history.R2Adjusted = 1 - (1-history.R2)*(m-1)/(m-n-1);
            history.w = ws';
            
            %% ���Լ����
            [examples,features] = size(testExamples);
            xx = testExamples(:,1:features-1);
            yy = testExamples(:,features);
            yyTest = xx*ws';
            history.testMSE = rssError(yy,yyTest)/m;
            %% R2ֵ
            history.testR2 = 1 - rssError(yy,yyTest)/sum((yy - mean(yy)).^2);
            %% R2Adjusted
            history.testR2Adjusted = 1 - (1-history.testR2)*(examples-1)/(examples-features-1);
end

function loss = Objective(x, w, y, lambda,mu)
    loss = ( 1/2 * sum((x*w - y).^2) + lambda*norm(w,1)+0.5*mu*norm(w,2) );
end