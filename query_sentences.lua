stringx = require('pl.stringx')
require 'io'
require 'nngraph'
require 'base'
ptb = require('data')

function readline()
	local line = io.read("*line")
	if line == nil then error({code="EOF"}) end
  	line = stringx.split(line)
  	if tonumber(line[1]) == nil then error({code="init"}) end
  
	word_map = {}	
	--get index of input words in vocab_map
	for i = 2, #line do
		if ptb.vocab_map[line[i]] == nil then
			word_map[i-1] = ptb.vocab_map["<unk>"]
		else
			word_map[i-1] = ptb.vocab_map[line[i]]
		end
	end
	return line, word_map
end

batch_size = 20
print("=>..Loading model..")
local filename = "./models/model.net"
model = torch.load(filename)
state_train = {data=ptb.traindataset(batch_size)}
state_valid =  {data=ptb.validdataset(batch_size)}

while true do
  	print("Query: len word1 word2 etc")
  	local ok, line, word_map = pcall(readline)
  	if not ok then
  		if line.code == "EOF" then
      			break -- end loop
    		elseif line.code == "init" then
      			print("Start with a number")
    		else
      			print(line)
      			print("Failed, try again")
    		end
  	else
    		g_disable_dropout(model.rnns)
		g_replace_table(model.s[0], model.start_s)
	
		local given = #line -1 --length of sentence given
		local len = tonumber(line[1]) --length to be predicted
		
		for i = 1, len do
			--use last word in sentence to make prediction
			local x = torch.Tensor(batch_size):fill(word_map[given+i-1])
			local y = torch.Tensor(batch_size):fill(word_map[given+i-1])
			perp_tmp, model.s[1],pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        		local _, max_p = pred:max(2)
			max_p= max_p:squeeze(2)[1]
			table.insert(word_map, max_p)
			g_replace_table(model.s[0], model.s[1])	
		end	

		--print the output
		output = ""
		for j = 1, #word_map do
			output = output .. " " .. ptb.reverse_map[word_map[j]]
		end	
		print(output)
		io.write('\n')
		g_enable_dropout(model.rnns)
	end
end

