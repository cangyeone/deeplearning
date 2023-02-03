        encdoer_output, h_encoder = encoder(x.to(device), h0)
        # 解码器输入带开始标签的数据和编码器最终状态 
        h_decoder = torch.zeros(
            [2, 1, decoder.n_hidden]).to(device)
        crr_word = "B"
        for itr in range(50):
            x = torch.tensor([[word2id.get(crr_word, 0)]], dtype=torch.long)
            y, h_decoder = decoder(x.to(device), h_decoder, encdoer_output)
            pid = y.argmax(dim=2) 
            pid = pid.cpu().numpy()[0, 0] 
            crr_word = id2word.get(pid) 
            if crr_word == "E":break 
            outwords.append(crr_word)