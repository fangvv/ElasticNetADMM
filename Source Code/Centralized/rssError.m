%% rssError������Ҫ�����þ������
function [ error ] = rssError( y, yTest )
    yDis = y-yTest;%���
    [m,n] = size(yDis);
    %��ƽ��
    for i = 1:m
        yDis(i) = yDis(i)^2;
    end
    error = sum(yDis);%���к�
end