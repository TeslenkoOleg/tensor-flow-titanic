
const dfd = require("danfojs-node");
const tf = require("@tensorflow/tfjs-node");
const BARCH_SIZE = 32;
async function load_process_data() {
    let df = await dfd.readCSV("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")

    //A feature engineering: Extract all titles from names columns
    let title = df['Name'].apply((x) => { return x.split(".")[0] }).values;
    //replace in df
    df.addColumn("Name",  title, { inplace: true });

    //label Encode Name feature
    let encoder = new dfd.LabelEncoder()
    let cols = ["Sex", "Name"]
    cols.forEach(col => {
        encoder.fit(df[col])
        enc_val = encoder.transform(df[col])
        df.addColumn( col, enc_val, { inplace: true })
    })

    let Xtrain, ytrain;
    Xtrain = df.iloc({ columns: [`1:`] });
    ytrain = df['Survived'];

    df.head().print()

    // Standardize the data with MinMaxScaler
    let scaler = new dfd.MinMaxScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)

    return [Xtrain.tensor, ytrain.tensor] //return the data as tensors
}


function get_model() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [7], units: 124, activation: 'relu', kernelInitializer: 'leCunNormal' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.summary();

    return model
}

async function train() {
    const model = await get_model()
    const data = await load_process_data()
    const Xtrain = data[0];
    const ytrain = data[1];

    model.compile({
        // optimizer: "rmsprop",
        optimizer: "adam",
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    console.log("Training started....")
    await model.fit(Xtrain, ytrain,{
        batchSize: BARCH_SIZE,
        epochs: 50,
        //validationSplit: 0.2,
        callbacks:{
            onEpochEnd: async(epoch, logs)=>{
                // console.log(`EPOCH (${epoch + 1}): Train Accuracy: ${(logs.acc * 100).toFixed(2)},
                //                                      Val Accuracy:  ${(logs.val_acc * 100).toFixed(2)}\n`);
            }
        }
    });
    // Pclass, Name, Sex, Age, Siblings/Spouses Aboard, Parents/Children Aboard, Fare
    const predictionDied = [3, 0, 0, 22, 2, 0, 7.25]; //died
    const predictionSurvived = [1, 1, 1, 26, 1, 1, 67.9]; //survived
    const resultPredict = await predictSample(predictionSurvived, model);
};
async function predictSample(sample, model) {
    let result1 = model.predict(tf.tensor(sample, [1, sample.length])).arraySync();
    let result2 = model.predict(tf.tensor(sample, [1, sample.length]));
    result2.print();
    console.log('result1', result1);
    const numResult = result1[0][0];
    const roundedResult = parseFloat(numResult.toFixed(4));
    console.log('roundedResult', roundedResult);
    console.log(`\n\nPrediction: ${roundedResult > 0.5 ? "Survived" : "Died"}\n\n`);
}

train()
