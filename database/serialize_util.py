import io
import gzip
import pickle
import hashlib

from transformator.transformator import to_detailed_matrix


def gen_qubo_key(Q, *args):
    """Hashes the QUBO using SHA-256 and returns the hexdigest as a string.

    Useful to uniquely identify a QUBO.
    """
    data = Q.tobytes()
    for arg in args:
        data += arg
    return hashlib.sha256(data).hexdigest().encode('ascii')


def deserialize_metadata(bytes_):
    """Deserializes bytes to a Metadata object.

    Since the database saves the QUBOs in a compact matter, a conversion back
    to the detailed view is performed using
    ``tooquo.transformator.transformator.to_detailed_matrix``.

    Returns the Metadata object.
    """
    bytes_ = io.BytesIO(bytes_)
    f = gzip.GzipFile(fileobj=bytes_, mode='rb')
    data = f.read()
    metadata = pickle.loads(data)
    f.close()

    metadata.Q = to_detailed_matrix(metadata.Q, metadata.qubo_size)
    metadata.Q_prime = to_detailed_matrix(metadata.Q_prime, metadata.qubo_size)
    metadata.recommendations = []
    return metadata
