import requests
import numpy as np
from flask import Flask, request
import pickle
import librosa
import pandas as pd
import warnings
import os 

  
# convert mp3 to wav file

ALLOWED_EXTENSIONS = set(['wav'])


app = Flask(__name__)


def downloadFile(url):
    response = requests.get(url)
    file_name, file_extension = os.path.splitext(url)
    if file_extension == ".mp3":
        open("./test.mp3", "wb").write(response.content)
        return file_extension
    open("test.wav", "wb").write(response.content)
    return file_extension


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def most_frequent(List):
    return max(set(List), key=List.count)


def predict(file):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # file = request.files['file']
    if file and allowed_file(file.filename):
        audio_data, sr = librosa.load(file)  # , offset=0, duration=30)
        audio_data, _ = librosa.effects.trim(audio_data)
        audio_data = audio_data[:661500]
        collection = np.split(audio_data, 10)

        audio_data = collection[0]

        d = librosa.feature.mfcc(
            np.array(audio_data).flatten(), sr=22050, n_mfcc=20)  # 36565
        d_var = d.var(axis=1).tolist()
        d_mean = d.mean(axis=1).tolist()
        test_data = []  # [d_var + d_mean]
        for i in range(20):
            test_data.append(d_mean[i])
            test_data.append(d_var[i])
            mfcc_names = []
        for i in range(1, 21):
            mfcc_str = "mfcc"+str(i)+"_mean"
            mfcc_names.append(mfcc_str)
            mfcc_str = "mfcc"+str(i)+"_var"
            mfcc_names.append(mfcc_str)
        test_frame = pd.DataFrame([test_data], columns=mfcc_names)
        test_data = []
        mfcc_names = []
        # chroma
        S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        # chroma_stft_mean
        chroma_mean = round(np.mean(chroma), 6)
        test_data.append(chroma_mean)
        # chrome_stft_var
        chroma_var = round(np.var(chroma), 6)
        test_data.append(chroma_var)
        # chroma_label
        mfcc_names.append("chroma_stft_mean")
        mfcc_names.append("chroma_stft_var")

        # rms
        rms = librosa.feature.rms(y=audio_data)
        # rms_mean
        rms_mean = round(np.mean(rms), 6)
        test_data.append(rms_mean)
        # rms_var
        rms_var = round(np.var(rms), 6)
        test_data.append(rms_var)
        # rms_label
        mfcc_names.append("rms_mean")
        mfcc_names.append("rms_var")

        # spectral_centroid
        cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        # spectral_centroid_mean
        sc_mean = round(np.mean(cent), 6)
        test_data.append(sc_mean)
        # spectral_centroid_var
        sc_var = round(np.var(cent), 6)
        test_data.append(sc_var)
        # sc_label
        mfcc_names.append("spectral_centroid_mean")
        mfcc_names.append("spectral_centroid_var")

        # spectral_bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        # spectral_bandwidth_mean
        spec_bw_mean = round(np.mean(spec_bw), 6)
        test_data.append(spec_bw_mean)
        # spectral_bandwidth_var
        spec_bw_var = round(np.var(spec_bw), 6)
        test_data.append(spec_bw_var)
        # sb_label
        mfcc_names.append("spectral_bandwidth_mean")
        mfcc_names.append("spectral_bandwidth_var")

        # rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        # rolloff_mean
        rolloff_mean = round(np.mean(rolloff), 6)
        test_data.append(rolloff_mean)
        # rolloff_var
        rolloff_var = round(np.var(rolloff), 6)
        test_data.append(rolloff_var)
        # rolloff_label
        mfcc_names.append("rolloff_mean")
        mfcc_names.append("rolloff_var")

        # zero_crossing_rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        # zero_crossing_rate_mean
        zcr_mean = round(np.mean(zcr), 6)
        test_data.append(zcr_mean)
        # zero_crossing_rate_var
        zcr_var = round(np.var(zcr), 6)
        test_data.append(zcr_var)
        # zero_crossing_rate_label
        mfcc_names.append("zero_crossing_rate_mean")
        mfcc_names.append("zero_crossing_rate_var")

        # harmony
        y = librosa.effects.harmonic(audio_data)
        harmony = librosa.feature.tonnetz(y=y, sr=sr)
        # harmony_mean
        harmony_mean = round(np.mean(harmony), 6)
        test_data.append(harmony_mean)
        # harmony_var
        harmony_var = round(np.var(harmony), 6)
        test_data.append(harmony_var)
        # harmony_label
        mfcc_names.append("harmony_mean")
        mfcc_names.append("harmony_var")

        # perceptr_mean
        # perceptr_var

        # tempo
        hop_length = 512
        oenv = librosa.onset.onset_strength(
            y=audio_data, sr=sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                    hop_length=hop_length)[0]

        tempo = round(tempo, 6)
        test_data.append(tempo)
        # tempo_label
        mfcc_names.append("tempo")
        d_var = d.var(axis=1).tolist()
        d_mean = d.mean(axis=1).tolist()
        # test_data = []#[d_var + d_mean]
        for i in range(20):
            test_data.append(d_mean[i])
            test_data.append(d_var[i])
        for i in range(1, 21):
            mfcc_str = "mfcc"+str(i)+"_mean"
            mfcc_names.append(mfcc_str)
            mfcc_str = "mfcc"+str(i)+"_var"
            mfcc_names.append(mfcc_str)

        scaler = pickle.load(open('./pickle/scalar.pkl', 'rb'))
        X_train = pickle.load(open('./pickle/xtrain.pkl', 'rb'))
        perm_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'mfcc1_mean', 'rolloff_mean', 'zero_crossing_rate_mean', 'perceptr_var', 'mfcc3_mean', 'rms_mean', 'chroma_stft_mean', 'mfcc2_mean', 'mfcc4_mean', 'mfcc9_mean', 'spectral_centroid_var', 'mfcc6_mean',
                            'rms_var', 'mfcc17_mean', 'spectral_bandwidth_var', 'mfcc11_mean', 'zero_crossing_rate_var', 'mfcc7_mean', 'mfcc5_mean', 'mfcc8_mean', 'mfcc10_mean', 'mfcc12_mean', 'rolloff_var', 'mfcc13_mean', 'mfcc3_var', 'mfcc18_mean', 'mfcc5_var', 'chroma_stft_var']
        test_frame = pd.DataFrame([test_data], columns=mfcc_names)
        testing_frame = pd.DataFrame(scaler.transform(
            test_frame), columns=X_train.columns)
        shorter_testing_frame = testing_frame[perm_features]

        val = 1
        while (val <= 9):

            audio_data = collection[val]
            d = librosa.feature.mfcc(
                np.array(audio_data).flatten(), sr=22050, n_mfcc=20)  # 36565
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            test_data = []  # [d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
            mfcc_names = []
            for i in range(1, 21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)
            test_frame = pd.DataFrame([test_data], columns=mfcc_names)
            test_data = []
            mfcc_names = []
            # chroma
            S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            # chroma_stft_mean
            chroma_mean = round(np.mean(chroma), 6)
            test_data.append(chroma_mean)
            # chrome_stft_var
            chroma_var = round(np.var(chroma), 6)
            test_data.append(chroma_var)
            # chroma_label
            mfcc_names.append("chroma_stft_mean")
            mfcc_names.append("chroma_stft_var")

            # rms
            rms = librosa.feature.rms(y=audio_data)
            # rms_mean
            rms_mean = round(np.mean(rms), 6)
            test_data.append(rms_mean)
            # rms_var
            rms_var = round(np.var(rms), 6)
            test_data.append(rms_var)
            # rms_label
            mfcc_names.append("rms_mean")
            mfcc_names.append("rms_var")

            # spectral_centroid
            cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            # spectral_centroid_mean
            sc_mean = round(np.mean(cent), 6)
            test_data.append(sc_mean)
            # spectral_centroid_var
            sc_var = round(np.var(cent), 6)
            test_data.append(sc_var)
            # sc_label
            mfcc_names.append("spectral_centroid_mean")
            mfcc_names.append("spectral_centroid_var")

            # spectral_bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sr)
            # spectral_bandwidth_mean
            spec_bw_mean = round(np.mean(spec_bw), 6)
            test_data.append(spec_bw_mean)
            # spectral_bandwidth_var
            spec_bw_var = round(np.var(spec_bw), 6)
            test_data.append(spec_bw_var)
            # sb_label
            mfcc_names.append("spectral_bandwidth_mean")
            mfcc_names.append("spectral_bandwidth_var")

            # rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            # rolloff_mean
            rolloff_mean = round(np.mean(rolloff), 6)
            test_data.append(rolloff_mean)
            # rolloff_var
            rolloff_var = round(np.var(rolloff), 6)
            test_data.append(rolloff_var)
            # rolloff_label
            mfcc_names.append("rolloff_mean")
            mfcc_names.append("rolloff_var")

            # zero_crossing_rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            # zero_crossing_rate_mean
            zcr_mean = round(np.mean(zcr), 6)
            test_data.append(zcr_mean)
            # zero_crossing_rate_var
            zcr_var = round(np.var(zcr), 6)
            test_data.append(zcr_var)
            # zero_crossing_rate_label
            mfcc_names.append("zero_crossing_rate_mean")
            mfcc_names.append("zero_crossing_rate_var")

            # harmony
            y = librosa.effects.harmonic(audio_data)
            harmony = librosa.feature.tonnetz(y=y, sr=sr)
            # harmony_mean
            harmony_mean = round(np.mean(harmony), 6)
            test_data.append(harmony_mean)
            # harmony_var
            harmony_var = round(np.var(harmony), 6)
            test_data.append(harmony_var)
            # harmony_label
            mfcc_names.append("harmony_mean")
            mfcc_names.append("harmony_var")

            # perceptr_mean
            # perceptr_var

            # tempo
            hop_length = 512
            oenv = librosa.onset.onset_strength(
                y=audio_data, sr=sr, hop_length=hop_length)
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                        hop_length=hop_length)[0]

            tempo = round(tempo, 6)
            test_data.append(tempo)
            # tempo_label
            mfcc_names.append("tempo")
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            # test_data = []#[d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
            for i in range(1, 21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)

            test_frame2 = pd.DataFrame([test_data], columns=mfcc_names)
            testing_frame2 = pd.DataFrame(scaler.transform(
                test_frame2), columns=X_train.columns)
            shorter_testing_frame2 = testing_frame2[perm_features[:30]]
            df_test = pd.concat(
                [shorter_testing_frame, shorter_testing_frame2])
            shorter_testing_frame = df_test
            val += 1

        cbc = pickle.load(open('./pickle/cbc.pkl', 'rb'))
        xgbc = pickle.load(open('./pickle/cbc.pkl', 'rb'))
        # change xgbc later
        gbc = pickle.load(open('./pickle/gbc.pkl', 'rb'))
        abc = pickle.load(open('./pickle/abc.pkl', 'rb'))
        rfc = pickle.load(open('./pickle/rfc.pkl', 'rb'))
        lr = pickle.load(open('./pickle/lr.pkl', 'rb'))
        cls = pickle.load(open('./pickle/cls.pkl', 'rb'))

        # Testing Input Data
        #from collections import Counter
        result = []
        results = []
        models = {'Catboost': cbc, 'XGBoost': xgbc, 'Random Forest': rfc, 'Gradient Boosting': gbc,
                    'AdaBoost': abc,  'Linear Regression': lr, 'KNN': cls}  # 'Random Forest':rfc,
        key_list = list(models.keys())
        val_list = list(models.values())

        for model in models.values():
            #position = val_list.index(model)

            for i in range(10):
                test = model.predict(df_test[i:(i+1)])
                result.append(test)
            t = max(result, key=result.count)

            if t == [[0]] or t == [['blues']]:
                genre_detected = 'blues'
            elif t == [[1]] or t == [['pop']]:
                genre_detected = 'pop'
            elif t == [[2]] or t == [['jazz']]:
                genre_detected = 'jazz'
            elif t == [[3]] or t == [['reggae']]:
                genre_detected = 'reggae'
            elif t == [[4]] or t == [['metal']]:
                genre_detected = 'metal'
            elif t == [[5]] or t == [['disco']]:
                genre_detected = 'disco'
            elif t == [[6]] or t == [['classical']]:
                genre_detected = 'classical'
            elif t == [[7]] or t == [['hiphop']]:
                genre_detected = 'hiphop'
            elif t == [[8]] or t == [['rock']]:
                genre_detected = 'rock'
            else:
                genre_detected = 'country'

            result.append(genre_detected)
            results.append(genre_detected)

        result = result[:7]
        # return genre_detected
    return [results, key_list]
    

@app.route('/genre', methods=['POST'])
def genre():
    if request.method == 'POST':
        file = request.files['files']
        results = predict(file)
        print(results)
        return f"{most_frequent(results[0])}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
