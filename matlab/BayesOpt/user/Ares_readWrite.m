function y = Ares_readWrite(params,dir)
    while ~isfile(dir+"/MSBO_sample_request")
        pause(1)
        if isfile(dir+"/MSBO_good_night")
            delete(dir+"/MSBO_good_night")
            exit
        end
    end
    delete(dir+"/MSBO_sample_request");
    fileID = fopen(dir+"/MSBO_sample",'w+');
    for i = 1:length(params)
        fprintf(fileID,'%f,',params(i));
    end
    fclose(fileID);
    while (isfile(dir+"/MSBO_sample") || ~isfile(dir+"/MSBO_objective"))
        pause(1)
    end
    fileID = fopen(dir+"/MSBO_objective",'r+');
    y = fscanf(fileID,'%f');
    if y<0
        y = ((y+10)*10-10)*3;
    end
    disp(y)
    fclose(fileID);
    delete(dir+"/MSBO_objective")
end
