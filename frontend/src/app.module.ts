import { Module } from '@nestjs/common';
import { AccidentConsumerService } from './kafka/accident-consumer.service';
import { AccidentsGateway } from './websocket/accidents.gateway';

@Module({
  imports: [],
  controllers: [],
  providers: [
    AccidentConsumerService,
    AccidentsGateway
  ],
})
export class AppModule {}
