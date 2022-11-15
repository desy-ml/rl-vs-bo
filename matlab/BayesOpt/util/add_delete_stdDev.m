for i = 1:5
    Y=data{i,2};
    c=cellfun('isempty',Y);
    Y=Y(~c);
    D=data{i,4};
    D=D{1};
    D=D(:,end);
    Y{end}=Y{end}-2*D;
    data{i,2}=Y;
end
