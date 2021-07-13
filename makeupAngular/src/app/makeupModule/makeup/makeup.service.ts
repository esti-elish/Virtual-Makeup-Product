import { Injectable } from '@angular/core';
import { HttpClient} from '@angular/common/http'
import { Observable } from 'rxjs';
@Injectable()
export class MakeupService {

  constructor(private _http:HttpClient) {
    

   }
   getScheduleFromServer():Observable<String>{
     debugger
      return this._http.get<String>("/api/Values")
        }
}
