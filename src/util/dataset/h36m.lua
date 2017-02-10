local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 17
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
    self.flipRef = {{2,5},   {3,6},   {4,7},
                    {12,15}, {13,16}, {14,17}}
    -- Pairs of joints for drawing skeleton
    --[[self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}} ]]--

    local annot = {}
    local tags = {'index','imgname','center','scale','f','c','part_2D','part_3Dmono','istrain'} -- 'part_3D' (global world coordinate)
    local a = hdf5.open(paths.concat(projectDir,'data/h36m/annot.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.valid = allIdxs[annot.istrain:eq(0)]
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]
  
        opt.nValidImgs = opt.idxRef.valid:size(1)

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts2D = self.annot.part_2D[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    local depth = self.annot.part_3Dmono[idx][{{},3}]
  
    return pts, c, s
end

return M.Dataset

