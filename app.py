from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar artefactos entrenados
preprocessor = joblib.load('preprocessor.pkl')
model_rfr    = joblib.load('model_rfr.pkl')
model_mlp    = joblib.load('model_mlp.pkl')
app.logger.debug('Artefactos cargados')

@app.route('/')
def home():
    # Renderizamos el formulario HTML
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer datos numéricos del formulario
        qty = float(request.form['Quantity'])
        disc = float(request.form['Discount'])
        prof = float(request.form['Profit'])

        # Leer fechas y calcular shipping_delay
        order_date = pd.to_datetime(request.form['Order_Date'], dayfirst=False)
        ship_date  = pd.to_datetime(request.form['Ship_Date'],  dayfirst=False)
        shipping_delay = (ship_date - order_date).days

        # Calcular variables derivadas
        discount_effect  = disc * qty
        profit_per_unit  = prof / (qty + 1)

        # Leer variables categóricas
        subcat = request.form['Sub-Category']
        cat    = request.form['Category']
        seg    = request.form['Segment']

        # Crear DataFrame para preprocesar
        input_data = {
            'Quantity': [qty],
            'Discount': [disc],
            'Profit': [prof],
            'shipping_delay': [shipping_delay],
            'discount_effect': [discount_effect],
            'profit_per_unit': [profit_per_unit],
            'Sub-Category': [subcat],
            'Category': [cat],
            'Segment': [seg]
        }
        df_input = pd.DataFrame(input_data)
        app.logger.debug(f'DataFrame entrada: {df_input}')

        # Preprocesar y predecir
        X_proc = preprocessor.transform(df_input)
        pr_rfr = model_rfr.predict(X_proc)[0]
        pr_mlp = model_mlp.predict(X_proc)[0]
        app.logger.debug(f'Predicciones — RFR: {pr_rfr}, MLP: {pr_mlp}')

        return jsonify({
            'rfr_prediction': round(pr_rfr, 2),
            'mlp_prediction': round(pr_mlp, 2)
        })
    except Exception as e:
        app.logger.error(f'Error en predicción: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)