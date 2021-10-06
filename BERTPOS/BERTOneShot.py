def BERTPOSOneShot(epsilon, test_sentences = [["Welcome", "to", "Sesame", "Street"]]):
    '''

    :param epsilon: The epsilon cutoff for prediction (1-confidence)
    :param test_sentences: A vector of vector of words (vector of sentences split up)
    :return: The prediction intervals as a vector of vector of POS tags

    '''

    #Loading Calibration Scores and tag order from model 5
    calibration_scores = np.load("calib.npy")
    tags = set(np.load("tags.npy"))

    for i in range(len(test_sentences)):
        for ii in range(len(test_sentences[i])):
            test_sentences[i][ii] = (test_sentences[i][ii], ".")

    tag2int = {}
    int2tag = {}

    for i, tag in enumerate(sorted(tags)):
        tag2int[tag] = i+1
        int2tag[i+1] = tag

    tag2int['-PAD-'] = 0
    int2tag[0] = '-PAD-'

    n_tags = len(tag2int)
    list(tag2int)

    M = 100
    test_sentences = split(test_sentences, M)

    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    # Params for bert model and tokenization

    test_text = text_sequence(test_sentences)

    test_label= tag_sequence(test_sentences)

    tokenizer = create_tokenizer_from_hub_module()

    tokens_a = test_text[0]
    orig_to_tok_map = []
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)

    for token in tokens_a:
        tokens.extend(tokenizer.tokenize(token))
        orig_to_tok_map.append(len(tokens)-1) # # keep last piece of tokenized term -->> gives better results!
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)
    input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])

    test_examples = convert_text_to_examples(test_text, test_label)

    (test_input_ids, test_input_masks, test_segment_ids, test_labels_ids) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=M+2)

    test_labels = tf.keras.utils.to_categorical(test_labels_ids, num_classes=n_tags)

    model = build_model(M+2)

    initialize_vars(sess)

    #Loading weights from model 4
    model.load_weights('bert_tagger' + '4' + '.h5')

    flat_calib = calibration_scores

    flat_calib.sort()

    y_pred_test = []
    y_true_test = []
    for i in tqdm(range(int(len(test_input_ids)))):
        YP = model.predict([test_input_ids[range(i, i+1)], test_input_masks[range(i, i+1)], test_segment_ids[range(i, i+1)]], batch_size= 1)
        y_pred_test.append(YP[0])
        y_true_test.append(test_labels[range(i,i+1)].argmax(-1)[0])


    interval_preds = []
    for i in range(len(y_pred_test)):
        sentencepred = []
        for ii in range(len(y_pred_test[i])):
            if test_labels[i][ii][0] != 1:
                currpred = []
                for iii in range(191):
                    if y_pred_test[i][ii][iii] > epsilon:
                        currpred.append(int2tag[iii])
                sentencepred.append(currpred)
        interval_preds.append(sentencepred)

    return interval_preds
