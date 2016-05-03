require('nngraph')
require('base')
require 'xlua'
ptb = require('data')

local batch_size = 20
local layers = 2

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2*layers do
            model.start_s[d]:zero()
        end
    end
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        
	--xlua.progress(i, 100)
	xlua.progress(i, len-1)
	local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

function run_train()
    reset_state(state_train)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_train.data:size(1)

    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1,  (len - 1) do

        --xlua.progress(i, 3000)
        xlua.progress(i, len-1)
        local x = state_train.data[i]
        local y = state_train.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Valid perplexity : " .. g_f3(torch.exp(perp /(len - 1))))
    g_enable_dropout(model.rnns)
end

state_train = {data=ptb.traindataset(batch_size)}
state_valid =  {data=ptb.validdataset(batch_size)}
state_test =  {data=ptb.testdataset(batch_size)}

print("==> Loading model")
model = torch.load('models/model.net')
run_train()
--run_test()

