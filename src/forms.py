from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import SubmitField, FileField, IntegerField, FloatField, StringField, PasswordField, BooleanField, SelectField
from wtforms.validators import DataRequired, NumberRange, Optional


class RFForm(FlaskForm):
    X_train = FileField('Choose training dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    y_train = FileField('Choose target variable for training dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    X_test = FileField('Choose test dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    y_test = FileField('Choose target variable for test dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    n_estimators = IntegerField('Set the number of trees: ', validators=[Optional(), NumberRange(min=1, max=1000)], default=100)
    max_depth = IntegerField('Set the max depth of trees: ', validators=[Optional(), NumberRange(min=1, max=100)], default=7)
    feature_subsample_size = IntegerField('Set the size of feature subsample: ', validators=[Optional(), NumberRange(min=1)], default=None)
    submit = SubmitField('Fit!')

class GBForm(FlaskForm):
    X_train = FileField('Choose training dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    y_train = FileField('Choose target variable for training dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    X_test = FileField('Choose test dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    y_test = FileField('Choose target variable for test dataset: ', validators=[DataRequired(), FileAllowed(['csv'], 'Data must be in csv-format!')])
    n_estimators = IntegerField('Set the number of trees: ', validators=[Optional(), NumberRange(min=1, max=1000)], default=100)
    max_depth = IntegerField('Set the max depth of trees: ', validators=[Optional(), NumberRange(min=1, max=100)], default=7)
    feature_subsample_size = IntegerField('Set the size of feature subsample: ', validators=[Optional(), NumberRange(min=1)], default=None)
    learning_rate = FloatField('Set the learning rate: ', validators=[Optional(), NumberRange(min=0, max=1)], default=0.1)
    submit = SubmitField('Fit!')

class Form1(FlaskForm):
    score = FloatField('Score')
    submit = SubmitField('See predictions')

class Form2(FlaskForm):
    submit = SubmitField('Download X_train')

class Form3(FlaskForm):
    submit = SubmitField('Download y_train')

class Form4(FlaskForm):
    submit = SubmitField('Download X_test')

class Form5(FlaskForm):
    submit = SubmitField('Download y_test')

class Form6(FlaskForm):
    submit = SubmitField('See parameters')

class Form7(FlaskForm):
    submit = SubmitField('Download RMSE on each iteration of training set')

class Form8(FlaskForm):
    submit = SubmitField('Download RMSE on each iteration of test set')

class ParamsRFForm(FlaskForm):
    n_estimators = IntegerField('N estimators')
    max_depth = IntegerField('Max depth')
    feature_subsample_size = IntegerField('Feature subsample size')
    submit = SubmitField('Go back')

class ParamsGBForm(FlaskForm):
    n_estimators = IntegerField('N estimators')
    max_depth = IntegerField('Max depth')
    feature_subsample_size = IntegerField('Feature subsample size')
    learning_rate = FloatField('Learning rate')
    submit = SubmitField('Go back')


# class LoginForm(FlaskForm):
#     username = StringField('User name: ', validators=[DataRequired()])
#     password = PasswordField('Password', validators=[DataRequired(), Length(min=4, max=12)])
#     remember = BooleanField('Remember me on this computer')
#     submit = SubmitField('Entry')
