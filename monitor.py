"""The Monitor manages database access (e.g. saving QUBOs in the database) and
decides whether to use Kafka.
"""
import traceback

#from confluent_kafka import Producer
#from confluent_kafka import KafkaError
#from confluent_kafka.admin import AdminClient
#from confluent_kafka.admin import NewTopic


__pdoc__ = {}
__pdoc__["Producer"] = False
__pdoc__["KafkaError"] = False
__pdoc__["AdminClient"] = False
__pdoc__["NewTopic"] = False


class Monitor:
    def __init__(self, cfg, db):
        """Initialize a Monitor object.

        Args:
            cfg: The global configuration. See more in ``tooquo.config``.
            db: An object extending tooquo.database.Database.
        """
        self.cfg = cfg
        self.db = db
        self.kafka_enabled = cfg['kafka']['enabled']
        if self.kafka_enabled:
            kafka_cfg = {'bootstrap.servers': cfg['kafka']['hostname']}
            self.admin_client = AdminClient(kafka_cfg)
            self.p = Producer(kafka_cfg)

            try:
                ret = self.admin_client.create_topics([NewTopic("tooquo", 1)])
                ret['tooquo'].result()
            except Exception as ex:
                if ex.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                    traceback.print_exc()

    def save(self, metadata):
        """Save a Metadata object in the database.
        """
        if self.kafka_enabled:
            metadata.kafka_cfg_id = self.cfg['pipeline']['cfg_id']
            data = metadata.serialize().read()
            self.p.poll(0)
            self.p.produce('tooquo', data)
            self.p.flush()
        else:
            self.db.save_metadata(metadata)
