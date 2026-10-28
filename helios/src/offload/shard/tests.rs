use super::*;

#[test]
    fn ranks_without_a_store_exchange_no_record_settings() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let zero_stream = RankStream::Tcp(server);
        let mut peer_stream = RankStream::Tcp(client);
        let zero = std::thread::spawn(move || {
            let mut peers = vec![zero_stream];
            join_rank_zero(None, &mut peers)
        });
        assert!(join_peer(None, &mut peer_stream, 1, 2).unwrap().is_none());
        assert!(zero.join().unwrap().unwrap().is_none());
    }
