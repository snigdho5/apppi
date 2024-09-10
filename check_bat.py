from flask import Flask, jsonify, render_template
import psutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('batview.html')

@app.route('/text-to-qr')
def text_to_qr():
    return render_template('txt2qr.html')

@app.route('/guess')
def guess():
    return render_template('guess.html')


@app.route('/battery')
def get_battery():
    battery = psutil.sensors_battery()
    if battery:
        percent = battery.percent
        power_plugged = battery.power_plugged
        return jsonify({
            'percent': percent,
            'plugged': power_plugged
        })
    else:
        return jsonify({'error': 'Battery information not available'})

if __name__ == '__main__':
    app.run(debug=True, port=8443)
