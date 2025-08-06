import { WebSocketGateway, WebSocketServer } from '@nestjs/websockets';
import { Server } from 'socket.io';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class AccidentsGateway {
  @WebSocketServer()
  server: Server;

  sendAccidentUpdate(accidentData: any) {
    console.log('Sending accident update:', accidentData);
    this.server.emit('accidentUpdate', accidentData);
    console.log('Accident update sent');
  }
}
