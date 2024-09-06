
from moviemain.interface.main import train

def load_model(epochs=5):
    """
    Load or train the model and return it.
    """

    # not sure what s1, s2 represent
    model, s1, s2, movies = train(epochs=epochs)
    return model, movies
