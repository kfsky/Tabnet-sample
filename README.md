# Tabnet-sample

## Tabnetとは？

## 実装方法
### install, import
```commandline
!pip install ../input/tabnet/pytorch_tabnet-2.0.1-py3-none-any.whl
```

その後に、下記のようにimportすればOK。
```python
from pytorch_tabnet.pretraining import TabNetPretrainer

from pytorch_tabnet.tab_model import TabnetRegressor
from pytorch_tabnet.tab_model import TabNetClassifier
```

### 学習データ・テストデータ
基本的にlightGBMで通している方法（BaseBlock）で問題なし。（最後にndarryにする点も）
ただし、引数次第で正規化する実装をしていた。（TabNetは不要）
```python
# 引数次第で正規化（TabNetは不要だがやった方が良いことも）してndarray（X_train, y_train, X_test）を返す
def get_tabnet_dateset(input_df, rankgauss=False, seed=0):
    """
    Args:
        input_df(pd.DataFrame): 前処理後のtrain_test
        rankgauss(bool): Trueだと数値列にrankgaussかける
        seed(int): seed
    Requirements: 以下の変数が外側で定義されていないと動かない
        - train_raw / test_rawという名前のtrainとtestの長さを保ったpd.DataFrame（切り分けに使用）
        - num_colsという名前の数値列名を格納したリスト（rankgaussをかける列の判別に使用）
        - train_rawに含まれるtargetの列名をTARGET_COLとして定義（y_trainの取得に使用）
    """
    if rankgauss:
        transformer = QuantileTransformer(n_quantiles=100, random_state=seed, output_distribution='normal')

        # pytorch-tabnetでカテゴリ変数として扱うもの以外を正規化
        # いったんすべてRankGaussをかけて数値列だけ入れ替える
        df_scaled = pd.DataFrame(transformer.fit_transform(input_df))
        df_scaled.columns = input_df.columns

        for c in list(df_scaled):
            if c not in num_cols:
                df_scaled[c] = input_df[c]

        X_train = df_scaled[:len(train_raw)].reset_index(drop=True).values
        X_test = df_scaled[len(train_raw):].reset_index(drop=True).values
        y_train = train_raw[TARGET_COL].values
         
    else:
        X_train = input_df[:len(train_raw)].reset_index(drop=True).values
        X_test = input_df[len(train_raw):].reset_index(drop=True).values
        y_train = train_raw[TARGET_COL].values

    return X_train, X_test, y_train
```

### カテゴリ変数について
カテゴリ変数について、その特徴量とユニーク数を設定するところが、pretrainの引数にある。
```python
# preprocess
def tabnet_preprocess(input_df, n_as_cat):
    """
    Args:
        input_df: train_test
        n_as_cat: unique数がこの数以下のnumericはcategoryとして扱う

    """
    output_df = input_df.copy()
    nunique = output_df.nunique()
    types = output_df.dtypes
    categorical_columns = []
    categorical_dims =  {}

    for i, col in enumerate(output_df.columns):
        if (types[col] == 'object')|(nunique[col] < n_as_cat):
            print(i, col, output_df[col].nunique())
            enc = LabelEncoder()
            output_df[col] = output_df[col].fillna("<missing>").astype('str')
            output_df[col] = enc.fit_transform(output_df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = output_df[col].nunique()
        else:
            train_indices = train_raw.index
            output_df.fillna(input_df.loc[train_indices, col].mean(), inplace=True)

    return output_df, categorical_columns, categorical_dims
```

この設定の理由は調査中

### 学習
基本的に、lightGBMの実装方法と同じように実装できる。以下は、予測まで込みなので、その点は少し違う
```python
# 学習・予測
def run_tabnet(cv, X_train, y_train, X_test,  tabnet_params, fit_params, pretrain=True):
  """
  Args:
    cv(list): バリデーションのリスト
    X_train(np.ndarray): 学習データ
    y_train(np.ndarray): 正解ラベル
    X_test(np.ndarray): 評価データ
    pretrain(bool): 事前学習をするかどうか
    tabnet_params: 学習パラメータ
    fit_params: パラメータ
    他はpytorch-tabnetのパラメータのうち、変更することがありそうなもの
  """

  oof = np.zeros((len(X_train),))
  test_preds_all = np.zeros((len(X_test),))
  models = []
  pretrainer = None

  if pretrain:
      pretrainer = TabNetPretrainer(**tabnet_params)
      print('★'*20, 'START PRETRAINING', '★'*20)
      pretrainer.fit(X_train=X_train,
                      eval_set=[X_train],
                      max_epochs=fit_params["max_epochs"],
                      patience=fit_params["patience"],
                      batch_size=fit_params["batch_size"],
                      virtual_batch_size=fit_params["virtual_batch_size"],
                      num_workers=fit_params["num_workers"],
                      pretraining_ratio=fit_params["pretraining_ratio"])
      print('★'*20, 'FINISH PRETRAINING', '★'*20)

  for fold, (tr_idx, va_idx) in enumerate(cv):
      print()
      print('FOLD: ', fold)

      x_tr, y_tr = X_train[tr_idx], y_train[tr_idx].reshape(-1, 1) # reshape必要
      x_va, y_va = X_train[va_idx], y_train[va_idx].reshape(-1, 1) # reshape必要
      
      model = TabNetRegressor(**tabnet_params)
      model.fit(X_train=x_tr,
                y_train=y_tr,
                eval_set=[(x_tr, y_tr),(x_va, y_va)],
                eval_name=['train', 'valid'],
                eval_metric=fit_params["eval_metric"],
                loss_fn=fit_params["loss_fn"],
                max_epochs=fit_params["max_epochs"],
                patience=fit_params["patience"],
                batch_size=fit_params["batch_size"],
                virtual_batch_size=fit_params["virtual_batch_size"],
                num_workers=fit_params["num_workers"],
                from_unsupervised=pretrainer)

      models.append(model)
      oof[va_idx] = model.predict(x_va).ravel()
      pred = model.predict(X_test).ravel()
      test_preds_all += pred / len(cv)

  score = mean_squared_error(y_train, oof) ** .5
  print('Whole RMSLE: {:.4f}'.format(score))

  return models, oof, test_preds_all
```

### 実行時間
atmaCup10thのサンプルコードで測定。GPUはV-100。
* pretrainなし：11min 24sec
* pretrainあり：16min01sec

データだったり、パラメータ等で一概に言えないが、基本的に、lightGBMと大きな変化はないのでは？という感じ。
ただし、GPUの条件にもよるのでローカル環境で手軽に行うには準備が必要そう。

### 可視化
ここも基本的にlightGBMと同じ。
```python
# importanceの可視化
def visualize_tabnet_importance(models, feat_train_df):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * .25),10))
    sns.boxenplot(data=feature_importance_df, 
                  y='feature_importance', 
                  x='column', 
                  order=order, 
                  ax=ax, 
                  palette='viridis', 
                  #orient='h'
                  )
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('TabNet Importance')
    ax.grid()
    fig.tight_layout()
    return fig, ax
```

maskの可視化もできる（ただし、特徴量が多すぎるとわかりにくい
```python
# maskの可視化
def visualize_mask(models_tabnet, feat_train_df):
    masks_list = []
    for model in models_tabnet:
        _, masks = model.explain(X_test)
        masks_list.append(masks)

    masks_dict = {}
    for i in range(len(masks_list[0])):
        masks_dict[i] = np.zeros(masks_list[0][0].shape)

    for masks in masks_list:
        for i in range(len(masks)):
            masks_dict[i] += masks[i] / len(masks_list)

    h = (len(masks_dict) + 2) // 3
    fig, axes = plt.subplots(h, 3, figsize=(20, h*3), sharey=True, sharex='col')

    for i in range(len(masks_dict)):
        if h == 1:
            sns.heatmap(masks_dict[i], cmap='viridis', xticklabels=feat_train_df.columns, ax=axes[i], cbar=False)
            axes[i].set_title(f'mask {i}')
            axes[i].yaxis.set_visible(False)
        else:
            sns.heatmap(masks_dict[i], cmap='viridis', xticklabels=feat_train_df.columns, ax=axes[i//3][i%3], cbar=False)
            axes[i//3][i%3].set_title(f'mask {i}')
            axes[i//3][i%3].yaxis.set_visible(False)

    return fig
```