import { Injectable, OnModuleInit } from '@nestjs/common';
import { Consumer, Kafka } from 'kafkajs';
import { AccidentsGateway } from '../websocket/accidents.gateway';

@Injectable()
export class AccidentConsumerService implements OnModuleInit {
  private readonly kafka: Kafka;
  private readonly consumer: Consumer;

  constructor(private readonly accidentsGateway: AccidentsGateway) {
    this.kafka = new Kafka({
      clientId: 'accident-location-consumer',
      brokers: [process.env.KAFKA_BROKER || 'localhost:9092'],
    });

    this.consumer = this.kafka.consumer({ groupId: 'accident-location-group' });
  }

  async onModuleInit() {
    await this.connect();
  }

  private async connect() {
    await this.consumer.connect();
    await this.consumer.subscribe({ topic: 'output_topic', fromBeginning: true });

    await this.consumer.run({
      eachMessage: async ({ message }) => {
        try {
          const accidentData = JSON.parse(message.value.toString());
          this.accidentsGateway.sendAccidentUpdate(accidentData);
        } catch (error) {
          console.error('Error processing message:', error);
        }
      },
    });
  }

  async disconnect() {
    await this.consumer.disconnect();
  }
}
