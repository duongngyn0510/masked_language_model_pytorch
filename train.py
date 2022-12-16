import torch


def train(model, n_tokens, criterion, optimizer, train_loader, epochs, device):

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_train = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input = batch['input_ids'].clone()
            src_mask = model.generate_square_subsequent_mask(batch['input_ids'].size(1))

            rand_value = torch.rand(batch['input_ids'].shape)
            rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0) # idx2token: {0: 'PAD', 101: 'SOS', 102: 'EOS'}
            mask_idx = rand_mask.flatten().nonzero().view(-1)

            input = input.flatten()
            input[mask_idx] = 103 # idx2token: {103: [MASK]}  
            input = input.view(batch['input_ids'].size())

            outputs = model(input.to(device), src_mask.to(device))
            loss = criterion(outputs.view(-1, n_tokens), batch['input_ids'].view(-1).to(device))

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch['input_ids'].size(0)
            n_train += batch['input_ids'].size(0)

        train_loss = train_loss / n_train

        if (epoch + 1) % 40 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')