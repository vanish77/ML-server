from flask import Flask, render_template, url_for, redirect, request, send_file
from flask_bootstrap import Bootstrap
from sklearn.metrics import mean_squared_error
import os
import ensembles
import pandas as pd
from forms import RFForm, GBForm, ParamsRFForm, ParamsGBForm, Form1, Form2, Form3, Form4, Form5, Form6, Form7, Form8

score = None
rmse_val = None
rmse_train = None
n_estimators = None
max_depth = None
feature_subsample_size = None
learning_rate = None
name = None
y_pred = None
UPLOAD_FOLDER = '/Users/ivanevgenyevich/MyProjects/Ensembles_Server/scripts/datasets'

app = Flask(__name__, template_folder = 'templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'apple'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/rf', methods=['GET', 'POST'])
def rf():
    global score, name, rmse_train, rmse_val, max_depth, n_estimators, feature_subsample_size, y_pred
    rf = RFForm()
    name = 'rf'
    if request.method == 'POST' and rf.validate_on_submit():
        X_train = pd.read_csv(rf.X_train.data)
        X_test = pd.read_csv(rf.X_test.data)
        y_train = pd.read_csv(rf.y_train.data)
        y_test = pd.read_csv(rf.y_test.data)
        X_train.to_csv(os.path.join('./datasets', 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join('./datasets', 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join('./datasets', 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join('./datasets', 'y_test.csv'), index=False)
        if rf.feature_subsample_size.data is None:
            fsize = X_train.shape[1] // 3
        else:
            fsize = rf.feature_subsample_size.data
        if fsize > X_train.shape[1]:
            fsize = X_train.shape[1]
        model = ensembles.RandomForestMSE_n_estimators(n_estimators=rf.n_estimators.data, max_depth=rf.max_depth.data, feature_subsample_size=fsize, flag_val=1)
        try:
            X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().ravel(), y_test.to_numpy().ravel()
            rmse_train, rmse_val = model.fit(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)
        except Exception:
            return redirect(url_for('rf'))
        score = mean_squared_error(y_test, y_pred, squared=True)
        n_estimators = rf.n_estimators.data
        max_depth = rf.max_depth.data
        feature_subsample_size = fsize
        return redirect(url_for('predict'))
    return render_template('quick_form.html', form=rf)

@app.route('/gb', methods=['GET', 'POST'])
def gb():
    global score, name, rmse_train, rmse_val, learning_rate, max_depth, n_estimators, feature_subsample_size, y_pred
    gb = GBForm()
    name = 'gb'
    if request.method == 'POST' and gb.validate_on_submit():
        X_train = pd.read_csv(gb.X_train.data)
        X_test = pd.read_csv(gb.X_test.data)
        y_train = pd.read_csv(gb.y_train.data)
        y_test = pd.read_csv(gb.y_test.data)
        X_train.to_csv(os.path.join('./datasets', 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join('./datasets', 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join('./datasets', 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join('./datasets', 'y_test.csv'), index=False)
        if gb.feature_subsample_size.data is None:
            fsize = X_train.shape[1] // 3
        else:
            fsize = gb.feature_subsample_size.data
        if fsize > X_train.shape[1]:
            fsize = X_train.shape[1]
        if gb.learning_rate.data == 0:
            lr = 0.001
        else:
            lr = gb.learning_rate.data
        model = ensembles.GradientBoostingMSE_n_estimators(n_estimators=gb.n_estimators.data, max_depth=gb.max_depth.data, feature_subsample_size=fsize, learning_rate=lr, flag_val=1)
        try:
            X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().ravel(), y_test.to_numpy().ravel()
            rmse_train, rmse_val = model.fit(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)
        except Exception:
            return redirect(url_for('gb'))
        score = mean_squared_error(y_test, y_pred, squared=True)
        n_estimators = gb.n_estimators.data
        max_depth = gb.max_depth.data
        feature_subsample_size = fsize
        learning_rate = lr
        return redirect(url_for('predict'))
    return render_template('quick_form.html', form=gb)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global score, rmse_train, rmse_val, y_pred, name
    form1 = Form1()
    form2 = Form2()
    form3 = Form3()
    form4 = Form4()
    form5 = Form5()
    form6 = Form6()
    form7 = Form7()
    form8 = Form8()
    form1.score.data = score
    if request.method == 'POST' and form1.validate_on_submit() and request.form['submit'] == 'See predictions':
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(os.path.join('./datasets', 'y_pred.csv'), index=False)
        return send_file(os.path.join('./datasets', 'y_pred.csv'), as_attachment=True)
    if request.method == 'POST' and form2.validate_on_submit() and request.form['submit'] == 'Download X_train':
        return send_file(os.path.join('./datasets', 'X_train.csv'), as_attachment=True)
    if request.method == 'POST' and form3.validate_on_submit() and request.form['submit'] == 'Download y_train':
        return send_file(os.path.join('./datasets', 'y_train.csv'), as_attachment=True)
    if request.method == 'POST' and form4.validate_on_submit() and request.form['submit'] == 'Download X_test':
        return send_file(os.path.join('./datasets', 'X_test.csv'), as_attachment=True)
    if request.method == 'POST' and form5.validate_on_submit() and request.form['submit'] == 'Download y_test':
        return send_file(os.path.join('./datasets', 'y_test.csv'), as_attachment=True)
    if request.method == 'POST' and form6.validate_on_submit() and request.form['submit'] == 'See parameters' and name == 'rf':
        return redirect(url_for('rf_params'))
    if request.method == 'POST' and form6.validate_on_submit() and request.form['submit'] == 'See parameters' and name == 'gb':
        return redirect(url_for('gb_params'))
    if request.method == 'POST' and form7.validate_on_submit() and request.form['submit'] == 'Download RMSE on each iteration of training set':
        rmse_train = pd.DataFrame(rmse_train)
        rmse_train.to_csv(os.path.join('datasets', 'rmse_train.csv'))
        return send_file(os.path.join('datasets', 'rmse_train.csv'), as_attachment=True)
    if request.method == 'POST' and form8.validate_on_submit() and request.form['submit'] == 'Download RMSE on each iteration of test set':
        rmse_val = pd.DataFrame(rmse_val, columns=['RMSE'])
        rmse_val.to_csv(os.path.join('./datasets', 'rmse_val.csv'))
        return send_file(os.path.join('./datasets', 'rmse_val.csv'), as_attachment=True)
    return render_template('quick_form_predict.html', form1=form1, form2=form2, form3=form3, form4=form4, form5=form5, form6=form6, form7=form7, form8=form8)

@app.route('/rf_params', methods=['GET', 'POST'])
def rf_params():
    global n_estimators, max_depth, feature_subsample_size
    form = ParamsRFForm()
    form.n_estimators.data = n_estimators
    form.max_depth.data = max_depth
    form.feature_subsample_size.data = feature_subsample_size
    if request.method == 'POST' and form.validate_on_submit():
        if request.form['submit'] == 'Go back':
            return redirect(url_for('predict'))
    return render_template('quick_form.html', form=form)

@app.route('/gb_params', methods=['GET', 'POST'])
def gb_params():
    global n_estimators, max_depth, feature_subsample_size, learning_rate
    form = ParamsGBForm()
    form.n_estimators.data = n_estimators
    form.max_depth.data = max_depth
    form.feature_subsample_size.data = feature_subsample_size
    form.learning_rate.data = learning_rate
    if request.method == 'POST' and form.validate_on_submit():
        if request.form['submit'] == 'Go back':
            return redirect(url_for('predict'))
    return render_template('quick_form.html', form=form)
