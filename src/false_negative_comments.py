import pandas as pd


def get_false_negative_comment(X_test_raw, y_test, y_pred, column_name, label_cols):

    cols_index = label_cols.index(column_name)


    df_errors = pd.DataFrame({
        'comment': list(X_test_raw),
        'true_label': list(y_test[:, cols_index]),
        'predicted': list(y_pred[:, cols_index])
    })

    fn = df_errors[(df_errors['true_label'] == 1) & (df_errors['predicted'] == 0)]
    fp = df_errors[(df_errors['true_label'] == 0) & (df_errors['predicted'] == 1)]

    print(f"\n=== LABEL: {column_name} === ")

    print("\n=== FALSE NEGATIVE (bỏ sót) ===")
    print(fn['comment'].head(10).to_string())

    print("\n=== FALSE POSITIVE (báo nhầm) ===")
    print(fp['comment'].head(10).to_string())