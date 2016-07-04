% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 
% TLD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TLD is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TLD.  If not, see <http://www.gnu.org/licenses/>.

function run_TLD()

addpath(genpath('.')); init_workspace; 

ids = 6:8;
backgrounds = 1:2;
contexts = 1:5;
lights = 1:4;

for id=ids
    id3 = sprintf('%03d', id);
    for bg=backgrounds
        for ctx=contexts
            for light=lights
                begin = tic;
                
                file = [id3 '_' num2str(bg) '_' num2str(ctx) '_' num2str(light) '/'];
                inputPath = ['../images/gray/' file];
                dirOut = ['_output/' file ];
                
                fprintf('Vídeo %s... ', file);
                
                [initialFrame, initialRect] = getInitialFrame([id3 '_' num2str(bg) '_' num2str(ctx) '_' num2str(light)]);
                
                opt.source          = struct('camera',0,'input',inputPath,'bb0',[]); % camera/directory swith, directory_name, initial_bounding_box (if empty, it will be selected by the user)
                opt.output          = dirOut; mkdir(opt.output); % output directory that will contain bounding boxes + confidence

                min_win             = 24; % minimal size of the object's bounding box in the scanning grid, it may significantly influence speed of TLD, set it to minimal size of the object
                patchsize           = [15 15]; % size of normalized patch in the object detector, larger sizes increase discriminability, must be square
                fliplr              = 0; % if set to one, the model automatically learns mirrored versions of the object
                maxbbox             = 1; % fraction of evaluated bounding boxes in every frame, maxbox = 0 means detector is truned off, if you don't care about speed set it to 1
                update_detector     = 1; % online learning on/off, of 0 detector is trained only in the first frame and then remains fixed
                opt.plot            = struct('pex',0,'nex',0,'dt',0,'confidence',0,'target',0,'replace',0,'drawoutput',3,'draw',0,'pts',0,'help', 0,'patch_rescale',1,'save',0); 

                % Do-not-change -----------------------------------------------------------

                opt.model           = struct('min_win',min_win,'patchsize',patchsize,'fliplr',fliplr,'ncc_thesame',0.95,'valid',0.5,'num_trees',10,'num_features',13,'thr_fern',0.5,'thr_nn',0.65,'thr_nn_valid',0.7);
                opt.p_par_init      = struct('num_closest',10,'num_warps',20,'noise',5,'angle',20,'shift',0.02,'scale',0.02); % synthesis of positive examples during initialization
                opt.p_par_update    = struct('num_closest',10,'num_warps',10,'noise',5,'angle',10,'shift',0.02,'scale',0.02); % synthesis of positive examples during update
                opt.n_par           = struct('overlap',0.2,'num_patches',100); % negative examples initialization/update
                opt.tracker         = struct('occlusion',10);
                opt.control         = struct('maxbbox',maxbbox,'update_detector',update_detector,'drop_img',1,'repeat',1);


                % Run TLD -----------------------------------------------------------------
                %profile on;
                [bb,conf] = tldExample(opt, initialFrame, initialRect);
                %profile off;
                %profile viewer;
                bb = bb';
                conf = conf';

                % Save results ------------------------------------------------------------
                writePositions([opt.output, file(1:end-1), '-faceTLD.txt'], initialFrame, bb);
                disp('Results saved to ./_output.');
                
                fprintf('Done: %f min\n', toc(begin)/60);
            end
        end
    end
end

end

function writePositions(filename, initialFrame, patches)
    
    f = -1;
    while f == -1
        f = fopen(filename, 'w');
    end
    
    patches(isnan(patches)) = 0;
    
    patches(:,3) = patches(:,3) - patches(:,1);
    patches(:,4) = patches(:,4) - patches(:,2);
    
    for index = 1:initialFrame-1
        fprintf(f, '%010d %05d\n', index, 0);
    end
    
    for index = initialFrame:size(patches,1)
        patch = patches(index,:);                
        
        if sum(patch,2) > 0
            fprintf(f, '%010d %04d %04d %04d %04d\n', index, floor(patch));
        else
            fprintf(f, '%010d %04d %04d %04d %04d\n', index, [0 0 0 0]);
        end
    end
    
    fclose(f);
end

function [initialFrame, initialRect] = getInitialFrame(file)

    gt_f = fopen(['../gts/' file '_oki_gt.txt'], 'r');
    C = textscan(gt_f, '%d %d %d %d %d');        
    m = floor(cell2mat(C(2:end)) ./ 2);
    
    initialRect = double(m(1,:));
    
    %initialFrame = imread(['../images/gray/' file '/' file '-gray_0001.bmp']);
    
    initialFrame = 1;
    
end
                

