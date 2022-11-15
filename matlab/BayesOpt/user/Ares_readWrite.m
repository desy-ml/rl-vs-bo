function y = Ares_readWrite(params,dir)
    while ~isfile(dir+"/MSBO_request.txt")
        pause(1)
    end
    delete(dir+"/MSBO_request.txt");
    fileID = fopen(dir+"/MSBO_sample.txt",'w+');
    for i = 1:length(params)
        fprintf(fileID,'%f,',params(i));
    end
    fclose(fileID);
    while (isfile(dir+"/MSBO_sample.txt") || ~isfile(dir+"/MSBO_objective.txt"))
        pause(1)
    end
    fileID = fopen(dir+"/MSBO_objective.txt",'r+');
    y = fscanf(fileID,'%f');
    fclose(fileID);
    delete(dir+"/MSBO_objective.txt")
end
