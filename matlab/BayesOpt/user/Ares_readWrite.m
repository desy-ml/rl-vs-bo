function y = Ares_readWrite(params,dir)
    fprintf("MSBO Matlab is waiting for sample request\n")
    while ~isfile(dir+"/MSBO_sample_request")
        pause(1)
        if isfile(dir+"/MSBO_good_night")
            fprintf("MSBO Matlab is going to sleep\n")
            delete(dir+"/MSBO_good_night")
            exit
        end
    end
    delete(dir+"/MSBO_sample_request");
    fprintf("MSBO Matlab is answering with a new sample\n")
    fileID = fopen(dir+"/MSBO_sample",'w+');
    for i = 1:length(params)
        fprintf(fileID,'%f,',params(i));
    end
    fclose(fileID);
    fprintf("MSBO Matlab is waiting for a new objective file\n")
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
    fprintf("MSBO Matlab received objective file\n")
end
