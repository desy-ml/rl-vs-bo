function arr = readPAM(directory)
% read .npy data 
    f=fopen(directory,"r");
    str=fgetl(f);
    I_start = strfind(str,'(')+1;
    I_stop = strfind(str,')')-1;
    type_id = strfind(str,'<');
    type=str(type_id+1:type_id+2);
    substr = str(I_start:I_stop);
    shape_cell = split(substr,',');
    shape_cell = shape_cell(~cellfun("isempty",shape_cell));
    shape = zeros(1,length(shape_cell));
    for i = 1:length(shape_cell)
        shape(i) = str2double(shape_cell{i,1});
    end
    shape = flip(shape);
    
    switch type
        case 'u8'
            arr = fread(f,shape,"int64");
        case 'f8'
            arr = fread(f,shape,"float64");
        case 'f4'
            arr = fread(f,shape,"float32");
    end
    arr = arr';
    fclose(f);


