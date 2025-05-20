import numpy as np
import faiss
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer

# Claude API クライアントを初期化
client = Anthropic(api_key="YOUR-API-KEY")

# ユーザーが書いたレビューの例
reviews = [
    "バックパック旅行の魅力は大きい。遠回りする。",
    "人間愛別に適度的距離を求く書られた感動的な物語。",
    "想像力のあるディストピア小説だが、展開が速すぎている。",
    "感動、壮大な世界を織り込んだ古典的なラブストーリー。",
    "哲学的な本質のある壮大な海洋冒険譚。",
    "歴史の世界観検に優れ，ロマンスが盛りなせる魅惑的な物語。",
    "美しい描写だが、展開が予測可能。",
    "恋愛とアートを通じた詳細かつ感動的な作品。",
    "ギリシャ神話の新鮮な解釈だが、展開にもたつきがある。",
    "感動的な人間像と個人の成長を見事に描き出した傑作。",
    "まだありがちなロマンチック・ユートピアか、今回は熱帯の島が舞台。",
]

# TF-IDFベクトライザーを初期化（embeddings代わりに使用）
vectorizer = TfidfVectorizer()

# テキストのベクトル化（埋め込み代わり）
def get_embedding_local(text):
    text = text.replace("\n", " ")
    # テキストが訓練されていない場合、先にフィットしておく
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.fit(reviews + [text])
    
    # テキストをベクトル化
    vector = vectorizer.transform([text]).toarray()[0]
    return vector

# レビューのインデックスを作成する関数
def index_reviews(reviews):
    # 全てのレビューでvectorizerをフィットする
    vectorizer.fit(reviews)
    
    # レビューのベクトルを取得
    vectors = vectorizer.transform(reviews).toarray()
    
    # インデックスを作成
    d = vectors.shape[1]  # ベクトルの次元
    index = faiss.IndexFlatL2(d)
    
    # ベクトルをインデックスに追加
    index.add(vectors)
    
    return index

# クエリに関連するレビューを検索する関数
def retrieve_reviews(index, query, reviews, k=2):
    # クエリの埋め込みベクトルを取得
    query_vector = get_embedding_local(query)
    
    # ベクトルを2次元配列に変形してインデックスを検索
    query_vector = np.array(query_vector).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    
    return [reviews[i] for i in indices[0]]

# 評価予測関数
def predict_rating(book, related_reviews):
    reviews = "\n".join(related_reviews)
    
    prompt = f"""以下に本から読もうと考えている本です：\n{book}\n\n以下は関連する過去のレビューです：\n{reviews}\n\n1(最低)から5(最高)の評価で、私がこの本を楽しめる可能性はどのくらいですか？理由も説明してください。数字だけで回答してください。"""
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000,
        temperature=0.7
    )
    
    return response.content[0].text

# 実行部分（コメントアウトを外して実行）
print("ローカルベクトライズバージョンのRAGシステムを初期化中...")
try:
    # インデックスを作成
    print("レビューのインデックスを作成中...")
    index = index_reviews(reviews)
    print("インデックス作成完了!")
    
    # テスト用の書籍
    book = "アレックス・ガーランドによる「ザ・ビーチ」は、手つかずの楽園を追い求めるバックパッカーたちの利己主義と過剰的欲望を描くことで、バックパッカー文化を批評的に描いている作品である。"
    
    # 関連レビューを取得
    print("関連レビューを検索中...")
    related_reviews = retrieve_reviews(index, book, reviews)
    print(f"関連レビュー: {related_reviews}")
    
    # ここではAPIキーが設定されていると仮定して、Anthropicへのリクエスト部分をコメントアウトしています
    # 実際に実行する場合はAPIキーを設定し、コメントを外してください
 
    # 予測評価を取得
    print("Claudeを使って評価を予測中...")
    rating = predict_rating(book, related_reviews)
    print(f"予測される評価: {rating}")
    
    print("処理完了!")
except Exception as e:
    print(f"エラーが発生しました: {e}")