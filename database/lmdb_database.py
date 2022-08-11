import traceback

import lmdb

from paths import PATH_DB
from database.database import Database
from database.serialize_util import gen_qubo_key
from database.serialize_util import deserialize_metadata


def int_to_bytes(x):
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


class LmdbDatabase(Database):
    def __init__(self, cfg, db_path=None):
        """Create a LMDB instance.
        
        Args:
            cfg: The config dictionary, as loaded from JSON (see
                ``tooquo.config.load_cfg``)
            db_path: If given, this path to the LMDB path will be used. If
                this is set to None, a path is inferred from the dataset id.
        """
        super(Database, self).__init__()
        if db_path is None:
            dataset_id = cfg["pipeline"]["dataset_id"]
            db_path = PATH_DB + '%s.lmdb' % dataset_id
        self.db_path = db_path
        self.env = None
        self.main_db = None
        self.key_db = None

    def init(self):
        self.env = lmdb.open(self.db_path, map_size=int(5e10), max_dbs=10)
        # TODO: Log this as debug
        # with self.env.begin() as txn:
        #     cursor = txn.cursor()
        #     for key, value in cursor:
        #         print(key)
        self.main_db = self.env.open_db(key="main".encode())
        self.key_db = self.env.open_db(key="key".encode())

    def close(self):
        self.env.close()
        self.env = None
        self.main_db = None
        self.key_db = None

    def get_metadata_by_qubo(self, Q):
        """Get a Metadata object by QUBO Q (vector of flattened QUBO)."""
        return self[gen_qubo_key(Q)]

    def save_metadata(self, metadata):
        """Save a Metadata object from the Monitor.

        The original input is either a QUBO or a problem definition."""
        idx = int_to_bytes(self.size() + 1)

        with self.env.begin(write=True, db=self.main_db) as txn:
            # txn.put(metadata.key(), metadata.serialize().read())
            txn.put(idx, metadata.serialize().read())

        with self.env.begin(write=True, db=self.key_db) as txn:
            txn.put(metadata.key(), idx)

        return idx

    def iter_metadata(self):
        """Iterate through the dataset. Returns a generator.
        """
        with self.env.begin(db=self.main_db) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    yield key, deserialize_metadata(value)
                except GeneratorExit:
                    return
                except:
                    print("ERROR FOR", key)
                    traceback.print_exc()

    def size(self):
        with self.env.begin() as txn:
            return txn.stat(self.key_db)["entries"]

    def __getitem__(self, key):
        with self.env.begin(db=self.key_db) as txn:
            idx = txn.get(key)

        with self.env.begin(db=self.main_db) as txn:
            data = txn.get(idx)

        return deserialize_metadata(data)

    def get_keys(self):
        with self.env.begin(db=self.key_db) as txn:
            keys = list(txn.cursor().iternext(values=False))
        return keys

    def get_indices(self):
        with self.env.begin(db=self.main_db) as txn:
            keys = list(txn.cursor().iternext(values=False))
        return keys
