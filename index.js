const { spawn } = require('child_process')

const generateModel = index => new Promise((resolve, reject) => {
    const model = spawn('./target/release/fang_oost_cal_charts', [index, JSON.stringify({ constraints: {} })])
    let modelOutput = ''
    let modelErr = ''
    model.stdout.on('data', data => {
        modelOutput += data
    })
    model.stderr.on('data', data => {
        modelErr += data
    })
    model.on('close', code => {

        if (modelErr) {
            reject(modelErr)
        }
        else {
            resolve(modelOutput)
        }
    })
})

Promise.all([0, 1, 2, 3, 4, 5, 6, 7].map(generateModel))
    .then(res => console.log(res))
    .catch(err => {
        console.log(err)
    })
