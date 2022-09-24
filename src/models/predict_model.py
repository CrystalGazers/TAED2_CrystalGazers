# Load model somehow

# 'El Periodico' test dataset
valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
y_pred = validate(model, None, test_x, None, params.batch_size, device)
y_token = [vocab.idx2token[index] for index in y_pred]

submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
print(submission.head())
submission.to_csv('submission.csv', index=False)