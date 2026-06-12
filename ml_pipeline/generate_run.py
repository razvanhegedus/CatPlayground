import matplotlib.pyplot as plt

# Your exact implementation string
code_text = """def train_model(train_loader, model, caju_loss, epochs = 30):    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    save_path = "best_catnet_model.pth"
    best_val_loss = float(1)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-6

        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            predictions = model(images)
            
            # Loss Calculation
            loss, hm_l, sz_l, off_l = caju_loss(predictions, targets)
            
            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss, hm_l, sz_l, off_l = caju_loss(outputs, targets)
                running_val_loss += loss.item() * images.size(0)
            
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        if is_loss_increasing(val_losses) == True:
            print(f"Early stopping triggered at epoch {epoch}! Loss has risen for 10 epochs straight.")
            break
    
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)"""

# Construct a matplotlib bounding box layout imitating an IDE window
fig, ax = plt.subplots(figsize=(10, 13), facecolor='#1e1e1e')
ax.set_facecolor('#1e1e1e')

# Render the text block inside the workspace coordinates
ax.text(0.02, 0.98, code_text, color='#d4d4d4', fontsize=11, fontfamily='monospace', 
        va='top', ha='left', transform=ax.transAxes)

# Strip out layout charts and borders
ax.axis('off')

# Save the final image asset cleanly to disk
output_path = "train_model_snippet.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), pad_inches=0.4)
plt.close()

print(f"Image successfully created and saved to: {output_path}")