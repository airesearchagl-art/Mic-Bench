このフォルダに faster-whisper 形式のモデルを配置してください。

フォルダ構成例:
  whisper_model/
    tiny/
      model.bin
      config.json
      tokenizer.json
      vocabulary.txt
      ...
    base/
      model.bin
      ...

モデルの入手方法:
  ネットワーク接続が可能な別環境で以下を実行し、生成されたフォルダをコピーしてください。

  from faster_whisper import WhisperModel
  WhisperModel("base", device="cpu", compute_type="int8")
  # キャッシュ先（Windows）: %USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-base
  # そのフォルダ内の snapshots\<hash>\ の中身を whisper_model/base/ にコピーする

対応モデルサイズ: tiny / base / small
