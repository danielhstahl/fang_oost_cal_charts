const {spawn} = require('child_process')
const results=require('./results.json')
results.options_and_rate.forEach(v=>{
    v.options=v.options.filter(({price})=>price>0)
    v.options.sort((a, b)=>a.strike-b.strike)
})

console.log(results.options_and_rate[2].options)
const generateModel=index=>new Promise((resolve, reject)=>{
    const model=spawn('./target/release/fang_oost_cal_charts', [index, JSON.stringify({...results, constraints:{}})])
    let modelOutput=''
    let modelErr=''
    model.stdout.on('data', data=>{
        modelOutput+=data
    })
    model.stderr.on('data', data=>{
        modelErr+=data
    })
    model.on('close', code=>{
        
        if(modelErr){
            reject(modelErr)
        }
        else {
            resolve(modelOutput)
        }
    })
})

Promise.all([0, 1, 2, 3].map(generateModel))
    .then(res=>console.log(res))
    .catch(err=>{
        console.log(err)
    })
