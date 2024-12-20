import random  # Para utilizar números aleatorios
import flask  # Para el servidor web
import pickle  # Para los pesos del modelo

def realiza_clasificacion(numeroEmbarazos, glucosa, presionArterial, grosorPiel, insulina, peso, altura, edad):
    imc = peso / (altura ** 2)
    datos = [numeroEmbarazos, glucosa, presionArterial, grosorPiel, insulina, imc, edad]
    prueba = [[abs(dato) for dato in datos]]

    clf = pickle.load(open("model/Final_model.sav", 'rb'))
    resultado = clf.predict(prueba)

    return resultado == 1  # Devuelve True si el modelo predice diabetes

# Crea el servidor Flask
servidor = flask.Flask(__name__)

@servidor.get("/")
def main():
    """
    Redirecciona a la página principal
    """
    return flask.redirect("/inicio")

@servidor.get("/inicio")
def muestra_inicio():
    """
    Muestra la página inicial
    """
    return flask.render_template("frontPage.html")

@servidor.get("/herramienta")
def muestra_herramienta():
    """
    Muestra la página de la herramienta
    """
    return flask.render_template("formulario.html")

@servidor.get("/resultado-saludable")
def muestra_resultado_saludable():
    """
    Muestra la página de resultados de la herramienta para un usuario saludable
    """
    return flask.render_template("resultado_saludable.html")

@servidor.get("/resultado-diabetes")
def muestra_resultado_diabetes():
    """
    Muestra la página de resultados de la herramienta para un usuario con diabetes
    """
    return flask.render_template("resultado_diabetes.html")

@servidor.route("/formulario.html", methods=["GET", "POST"])
def clasifica_usuario():
    """
    Muestra el formulario o procesa la clasificación según el método HTTP
    """
    if flask.request.method == "GET":
        # Si es una solicitud GET, muestra el formulario
        return flask.render_template("formulario.html")
    elif flask.request.method == "POST":
        # Si es una solicitud POST, procesa los datos del formulario
        numeroEmbarazos = int(flask.request.form["numero-embarazos"])
        glucosa = float(flask.request.form["glucosa"])
        presionArterial = float(flask.request.form["presion-arterial"])
        grosorPiel = float(flask.request.form["grosor-piel"])
        insulina = float(flask.request.form["insulina"])
        peso = float(flask.request.form["peso"])
        altura = float(flask.request.form["altura"])
        edad = int(flask.request.form["edad"])

        # Clasifica al usuario
        padeceDiabetes = realiza_clasificacion(numeroEmbarazos, glucosa, presionArterial, grosorPiel, insulina, peso, altura, edad)

        # Redirecciona al resultado correspondiente
        if padeceDiabetes:
            return flask.redirect("/resultado-diabetes")
        else:
            return flask.redirect("/resultado-saludable")

# Ejecuta el servidor
if __name__ == "__main__":
    servidor.run()
